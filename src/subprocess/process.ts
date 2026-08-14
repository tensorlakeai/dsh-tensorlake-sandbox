/** One asynchronously-started Tensorlake process projected onto the subprocess seam. */

import { Buffer } from 'node:buffer'
import { PassThrough, Writable } from 'node:stream'
import { posix } from 'node:path'
import { isRemoteMissing, OutputMode, ProcessStatus, RemoteAPIError, StdinMode } from '../runtime.ts'
import type { Sandbox } from '../runtime.ts'
import type {
  SubprocessHandle,
  SubprocessOutcome,
  SubprocessOutputMode,
  SubprocessSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'
import type TensorlakeRuntime from '../runtime.ts'
import { environmentArguments, readAmbientEnvironment } from './environment.ts'
import { OUTPUT_ENCODER_SOURCE, TensorlakeOutputDecoder, TensorlakeOutputReader } from './output.ts'
import { asError, buildProviderStdio, hasSpillMode, ProviderTermination, waitTick } from './provider.ts'
import { groupAlive, SHELL_NAME, signalRemoteGroups } from './remote.ts'

function isValidProcessId(value: number): boolean {
  return Number.isSafeInteger(value) && value > 1
}

interface RemotePaths {
  status: string
  stdoutPipe: string
  stderrPipe: string
  stdoutSpill: string
  stderrSpill: string
}

const PREPARE_SCRIPT = [
  'set -e',
  'mkdir -p -- "$1"',
  'chmod 700 -- "$1"',
  'mkfifo -- "$2" "$3"',
  'chmod 600 -- "$2" "$3"',
  'shift 3',
  'for dsh_file in "$@"; do : > "$dsh_file"; chmod 600 -- "$dsh_file"; done',
].join('\n')

/**
 * Build the wrapper the daemon starts. Positional arguments carry the private
 * state paths, the encoder source, the rebuilt environment, and the exact
 * argv, so no value ever crosses a shell-quoting layer. The encoders run as
 * explicit background jobs on named pipes — not process substitutions, whose
 * pids a bare `wait` does not cover — because the daemon stops capturing at
 * the direct process's exit: the wrapper publishes the user command's exit
 * status the moment it exits, then holds its own exit on both encoder pids so
 * every captured stream ends with its completion marker first.
 * @param spec - Fully resolved subprocess request.
 * @returns The bash script text.
 */
function wrapperScript(spec: SubprocessSpawnSpec): string {
  const encoder = (spill: SubprocessOutputMode, spillVariable: string): string => hasSpillMode(spill)
    ? `tee --output-error=warn-nopipe >(head -c ${spill.spill.maxBytes} > "${spillVariable}") | node -e "$dsh_encoder"`
    : 'node -e "$dsh_encoder"'
  return [
    'set +e',
    'dsh_status=$1',
    'dsh_out_pipe=$2',
    'dsh_err_pipe=$3',
    'dsh_out_spill=$4',
    'dsh_err_spill=$5',
    'dsh_encoder=$6',
    'dsh_envc=$7',
    'shift 7',
    'dsh_env=( "${@:1:dsh_envc}" )',
    'shift "$dsh_envc"',
    `{ ${encoder(spec.stdio.stdout, '$dsh_out_spill')}; } < "$dsh_out_pipe" &`,
    'dsh_out_encoder=$!',
    `{ ${encoder(spec.stdio.stderr, '$dsh_err_spill')}; } < "$dsh_err_pipe" >&2 &`,
    'dsh_err_encoder=$!',
    'env -i -- "${dsh_env[@]}" "$@" > "$dsh_out_pipe" 2> "$dsh_err_pipe"',
    'dsh_code=$?',
    'printf \'%s\\n\' "$dsh_code" > "$dsh_status"',
    'wait "$dsh_out_encoder" "$dsh_err_encoder"',
    'exit "$dsh_code"',
  ].join('\n')
}

class DeferredStdin extends Writable {
  constructor(private readonly ready: Promise<{ sandbox: Sandbox; pid: number }>) {
    super({ decodeStrings: false })
  }

  override _write(chunk: string | Buffer, _encoding: BufferEncoding, callback: (error?: Error | null) => void): void {
    void this.ready.then(({ sandbox, pid }) => sandbox.writeStdin(pid, Buffer.from(chunk))).then(
      () => { callback() },
      (error: unknown) => { callback(asError(error)) },
    )
  }

  override _final(callback: (error?: Error | null) => void): void {
    void this.ready.then(({ sandbox, pid }) => sandbox.closeStdin(pid)).then(
      () => { callback() },
      (error: unknown) => { callback(asError(error)) },
    )
  }
}

/** Tensorlake-backed subprocess handle rooted in one remote process group. */
export class TensorlakeSubprocessHandle implements SubprocessHandle {
  readonly stdin: Writable | undefined
  readonly stdout: PassThrough | undefined
  readonly stderr: PassThrough | undefined
  readonly collected: SubprocessHandle['collected']
  readonly done: Promise<SubprocessOutcome>

  private readonly startedState = Promise.withResolvers<number | undefined>()
  private readonly stdinState = Promise.withResolvers<{ sandbox: Sandbox; pid: number }>()
  private readonly stdoutDecoder = new TensorlakeOutputDecoder()
  private readonly stderrDecoder = new TensorlakeOutputDecoder()
  private readonly terminationController = new AbortController()
  private readonly followController = new AbortController()
  private readonly stdoutReader: TensorlakeOutputReader | undefined
  private readonly stderrReader: TensorlakeOutputReader | undefined
  private readonly paths: RemotePaths
  private readonly termination = new ProviderTermination()
  private remotePid = -1
  private outputTransportError: Error | undefined
  private outputDrainExpired = false
  private stateDirectoryCreated = false
  private terminationSignal: NodeJS.Signals | null = null

  /**
   * Begin a Tensorlake process without blocking the synchronous spawn call.
   * @param runtime - Shared Tensorlake sandbox owner.
   * @param spec - Fully resolved subprocess request.
   * @param stateDir - Remote directory retaining exit status and valid spills.
   * @param pollMs - Remote status/liveness poll cadence.
   */
  constructor(
    private readonly runtime: TensorlakeRuntime,
    private readonly spec: SubprocessSpawnSpec,
    readonly stateDir: string,
    private readonly pollMs: number,
  ) {
    this.paths = {
      status: posix.join(stateDir, 'exit-code'),
      stdoutPipe: posix.join(stateDir, 'stdout.pipe'),
      stderrPipe: posix.join(stateDir, 'stderr.pipe'),
      stdoutSpill: posix.join(stateDir, 'stdout.log'),
      stderrSpill: posix.join(stateDir, 'stderr.log'),
    }
    const { stdout, stderr, stdoutReader, stderrReader, collected } = buildProviderStdio(
      spec.stdio,
      (mode, stream) => new TensorlakeOutputReader(
        mode.maxBytes,
        mode.spill?.maxBytes,
        stream === 'stdout' ? this.paths.stdoutSpill : this.paths.stderrSpill,
      ),
    )
    this.stdout = stdout
    this.stderr = stderr
    this.stdoutReader = stdoutReader
    this.stderrReader = stderrReader
    this.collected = collected
    this.stdin = spec.stdio.stdin === 'pipe' ? new DeferredStdin(this.stdinState.promise) : undefined
    void this.stdinState.promise.catch(() => {})
    spec.signal?.addEventListener('abort', this.onAbort, { once: true })
    const execution = this.run()
    void execution.catch(() => {})
    this.done = execution
    if (spec.signal?.aborted === true) this.terminate()
  }

  /** @inheritdoc */
  terminate(): void {
    if (this.termination.proven || this.termination.pending !== undefined) return
    this.terminationController.abort(new Error('subprocess-tensorlake: command terminated'))
    this.followController.abort()
    this.stdout?.destroy()
    this.stderr?.destroy()
    this.termination.begin(() => this.terminateRemote())
  }

  /** @inheritdoc */
  async waitForExit(signal?: AbortSignal): Promise<boolean> {
    if (this.termination.proven) return true
    const started = await Promise.race([
      this.startedState.promise,
      new Promise<'aborted'>((resolve) => {
        if (signal === undefined) return
        if (signal.aborted) { resolve('aborted'); return }
        signal.addEventListener('abort', () => { resolve('aborted') }, { once: true })
      }),
    ])
    if (started === 'aborted') return false
    if (started === undefined) {
      this.termination.markQuiescent()
      return true
    }
    this.termination.throwFailure()
    let sandbox: Sandbox
    try {
      sandbox = await this.runtime.getSandbox()
    } catch (error: unknown) {
      if (signal?.aborted === true) return false
      if (isRemoteMissing(error)) {
        this.termination.markQuiescent()
        return true
      }
      throw error
    }
    return this.termination.awaitQuiescence(() => groupAlive(sandbox, started), this.pollMs, signal)
  }

  /** Remote process id after start; `-1` while startup is pending or after it fails. */
  get pid(): number {
    return this.remotePid
  }

  private readonly onAbort = (): void => { this.terminate() }

  private async run(): Promise<SubprocessOutcome> {
    let sandbox: Sandbox | undefined
    let preparing = true
    let follower: Promise<void> | undefined
    try {
      sandbox = await this.runtime.getSandbox()
      const ambient = await readAmbientEnvironment(sandbox)
      const envArgs = environmentArguments(ambient, this.spec.env)
      this.terminationController.signal.throwIfAborted()
      await this.prepareState(sandbox)
      this.terminationController.signal.throwIfAborted()
      preparing = false
      const info = await sandbox.startProcess('bash', {
        args: [
          '-c',
          wrapperScript(this.spec),
          SHELL_NAME,
          this.paths.status,
          this.paths.stdoutPipe,
          this.paths.stderrPipe,
          this.paths.stdoutSpill,
          this.paths.stderrSpill,
          OUTPUT_ENCODER_SOURCE,
          String(envArgs.length),
          ...envArgs,
          ...this.spec.argv,
        ],
        workingDir: this.spec.cwd,
        stdinMode: this.spec.stdio.stdin === 'ignore' ? StdinMode.CLOSED : StdinMode.PIPE,
        stdoutMode: OutputMode.CAPTURE,
        stderrMode: OutputMode.CAPTURE,
      })
      // The daemon starts the wrapper as its own session and process-group
      // leader, so the returned pid addresses the whole tree. Refuse ids whose
      // negative form addresses every process (`kill -- -1`) or init's group.
      if (!isValidProcessId(info.pid)) {
        await this.rollbackInvalidStart(sandbox, info.pid)
        throw new Error(`subprocess-tensorlake: Tensorlake returned unsafe process id ${info.pid}`)
      }
      this.remotePid = info.pid
      this.startedState.resolve(info.pid)
      this.stdinState.resolve({ sandbox, pid: info.pid })
      follower = this.followOutput(sandbox, info.pid)
      await this.writeBatchStdin(sandbox, info.pid)
      const published = await this.waitForStatus(sandbox, info.pid)
      await this.drainOutput(follower)
      if (this.outputTransportError !== undefined) throw this.outputTransportError
      const requireCompleteOutput = this.terminationSignal === null && !this.outputDrainExpired
      this.stdoutDecoder.finish(requireCompleteOutput)
      this.stderrDecoder.finish(requireCompleteOutput)
      await this.finalizeSpills(sandbox)
      if (published === undefined) {
        // Only a terminated wrapper may exit silently; waitForStatus threw otherwise.
        return { exitCode: null, signal: this.terminationSignal ?? 'SIGKILL' }
      }
      return this.terminationSignal === null
        ? { exitCode: published, signal: null }
        : { exitCode: null, signal: this.terminationSignal }
    } catch (error: unknown) {
      const canceledPreparation = preparing && this.terminationController.signal.aborted
      let failure = await this.termination.rollback(
        error,
        this.remotePid > 0,
        this,
        'subprocess-tensorlake: command monitoring failed and process-group rollback did not reach quiescence',
      )
      if (sandbox !== undefined && this.stateDirectoryCreated) {
        try {
          await this.removeFailedState(sandbox)
        } catch (cleanupError: unknown) {
          failure = new AggregateError(
            [failure, cleanupError],
            'subprocess-tensorlake: command failed and private state cleanup failed',
          )
        }
      }
      this.startedState.resolve(undefined)
      this.stdinState.reject(asError(failure))
      if (canceledPreparation && failure === error) return { exitCode: null, signal: 'SIGTERM' }
      throw failure
    } finally {
      this.followController.abort()
      await follower?.catch(() => {})
      this.spec.signal?.removeEventListener('abort', this.onAbort)
      this.stdout?.end()
      this.stderr?.end()
    }
  }

  private async rollbackInvalidStart(sandbox: Sandbox, pid: number): Promise<void> {
    try {
      await sandbox.killProcess(pid)
      this.termination.markQuiescent()
    } catch (error: unknown) {
      if (error instanceof RemoteAPIError) {
        // "not running" and kin: the daemon already reaped whatever it started.
        this.termination.markQuiescent()
        return
      }
      this.termination.recordFailure(error)
      throw asError(error)
    }
  }

  private async prepareState(sandbox: Sandbox): Promise<void> {
    // Own the directory before the request: a cancellation racing a committed
    // creation must still enter cleanup (removal tolerates an absent path).
    this.stateDirectoryCreated = true
    const files = [
      this.paths.status,
      ...(hasSpillMode(this.spec.stdio.stdout) ? [this.paths.stdoutSpill] : []),
      ...(hasSpillMode(this.spec.stdio.stderr) ? [this.paths.stderrSpill] : []),
    ]
    const result = await sandbox.run('bash', {
      args: ['-c', PREPARE_SCRIPT, SHELL_NAME, this.stateDir, this.paths.stdoutPipe, this.paths.stderrPipe, ...files],
    })
    if (result.exitCode !== 0) {
      throw new Error(`subprocess-tensorlake: private state setup exited ${result.exitCode}: ${result.stderr.trim()}`)
    }
  }

  private async writeBatchStdin(sandbox: Sandbox, pid: number): Promise<void> {
    if (typeof this.spec.stdio.stdin !== 'object') return
    try {
      await sandbox.writeStdin(pid, Buffer.from(this.spec.stdio.stdin.data, 'utf8'))
      await sandbox.closeStdin(pid)
    } catch {
      // Like the local adapter, batch stdin is best-effort; exit and output remain authoritative.
    }
  }

  private async followOutput(sandbox: Sandbox, pid: number): Promise<void> {
    try {
      for await (const event of sandbox.followOutput(pid, { signal: this.followController.signal })) {
        if (event.stream !== 'stdout' && event.stream !== 'stderr') {
          throw new Error(`subprocess-tensorlake: output transport delivered unknown stream ${JSON.stringify(event.stream)}`)
        }
        const stderr = event.stream === 'stderr'
        const bytes = stderr ? this.stderrDecoder.push(event.line) : this.stdoutDecoder.push(event.line)
        if (stderr) {
          this.stderrReader?.push(bytes)
          await this.deliver(this.stderr, this.spec.stdio.stderr === 'inherit' ? process.stderr : undefined, bytes)
        } else {
          this.stdoutReader?.push(bytes)
          await this.deliver(this.stdout, this.spec.stdio.stdout === 'inherit' ? process.stdout : undefined, bytes)
        }
      }
    } catch (error: unknown) {
      if (this.followController.signal.aborted) return
      this.outputTransportError ??= asError(error)
      this.stdout?.destroy(this.outputTransportError)
      this.stderr?.destroy(this.outputTransportError)
    }
  }

  private async deliver(
    pipe: PassThrough | undefined,
    inherited: NodeJS.WriteStream | undefined,
    bytes: Uint8Array,
  ): Promise<void> {
    const target = pipe ?? inherited
    if (target === undefined || bytes.length === 0 || this.terminationController.signal.aborted) return
    if (target.destroyed) throw new Error('subprocess output stream is closed')
    if (target.write(bytes)) return
    await new Promise<void>((resolve, reject) => {
      const settle = (): void => { cleanup(); resolve() }
      const onError = (error: Error): void => { cleanup(); reject(error) }
      const cleanup = (): void => {
        target.removeListener('drain', settle)
        target.removeListener('close', settle)
        target.removeListener('error', onError)
        this.terminationController.signal.removeEventListener('abort', settle)
        this.followController.signal.removeEventListener('abort', settle)
      }
      target.once('drain', settle)
      target.once('close', settle)
      target.once('error', onError)
      this.terminationController.signal.addEventListener('abort', settle, { once: true })
      this.followController.signal.addEventListener('abort', settle, { once: true })
      if (this.terminationController.signal.aborted || this.followController.signal.aborted) settle()
    })
  }

  private async waitForStatus(sandbox: Sandbox, pid: number): Promise<number | undefined> {
    for (;;) {
      this.termination.throwFailure()
      const raw = new TextDecoder().decode(await sandbox.readFile(this.paths.status)).trim()
      if (raw.length > 0) {
        const exitCode = Number(raw)
        if (!/^(?:0|[1-9][0-9]*)$/.test(raw) || !Number.isSafeInteger(exitCode) || exitCode > 255) {
          throw new Error(`subprocess-tensorlake: remote wrapper published invalid exit code ${JSON.stringify(raw)}`)
        }
        return exitCode
      }
      const info = await sandbox.getProcess(pid).catch((error: unknown) => {
        if (isRemoteMissing(error)) return undefined
        throw error
      })
      if (info === undefined || info.status !== ProcessStatus.RUNNING) {
        // One more read closes the exit-then-publish race before concluding silence.
        const lastRaw = new TextDecoder().decode(await sandbox.readFile(this.paths.status)).trim()
        if (lastRaw.length > 0) continue
        if (this.terminationSignal !== null || this.terminationController.signal.aborted) return undefined
        throw new Error('subprocess-tensorlake: remote wrapper exited before publishing its exit status')
      }
      await waitTick(this.pollMs)
    }
  }

  private async drainOutput(follower: Promise<void>): Promise<void> {
    const deadline = Date.now() + this.spec.graceMs
    while (!(this.stdoutDecoder.isComplete && this.stderrDecoder.isComplete)
      && this.outputTransportError === undefined
      && this.terminationSignal === null
      && !this.terminationController.signal.aborted) {
      if (Date.now() >= deadline) {
        // A descendant still holds an output pipe; the encoders cannot reach
        // EOF, so stop following and mark the spills incomplete.
        this.outputDrainExpired = true
        this.stdoutReader?.invalidateSpill()
        this.stderrReader?.invalidateSpill()
        break
      }
      await waitTick(this.pollMs)
    }
    this.followController.abort()
    await follower.catch(() => {})
  }

  private async terminateRemote(): Promise<void> {
    const pid = await this.startedState.promise
    if (pid === undefined) {
      this.termination.markQuiescent()
      return
    }
    let sandbox: Sandbox
    try {
      sandbox = await this.runtime.getSandbox()
    } catch (error: unknown) {
      if (isRemoteMissing(error)) {
        this.termination.markQuiescent()
        return
      }
      throw error
    }
    this.terminationSignal = 'SIGTERM'
    try {
      await signalRemoteGroups(sandbox, [pid], 'TERM')
      if (await this.waitForGroupExit(sandbox, pid)) {
        this.termination.markQuiescent()
        return
      }
    } catch {
      // Failed TERM delivery or observation cannot prove exit; force cleanup still owns the group.
    }
    this.terminationSignal = 'SIGKILL'
    try {
      await signalRemoteGroups(sandbox, [pid], 'KILL')
    } catch {
      // The daemon kill and the final liveness probe remain independent cleanup paths.
    }
    try {
      await sandbox.killProcess(pid)
    } catch {
      // The final liveness probe, not either transport's self-report, proves cleanup.
    }
    if (await this.waitForGroupExit(sandbox, pid)) {
      this.termination.markQuiescent()
      return
    }
    throw new Error(`subprocess-tensorlake: remote process group ${pid} remained live after force termination`)
  }

  private async waitForGroupExit(sandbox: Sandbox, processGroupId: number): Promise<boolean> {
    const deadline = Date.now() + this.spec.graceMs
    while (await groupAlive(sandbox, processGroupId)) {
      if (Date.now() >= deadline) return false
      await waitTick(this.pollMs)
    }
    return true
  }

  private async finalizeSpills(sandbox: Sandbox): Promise<void> {
    const removals: Promise<void>[] = []
    const collect = (mode: SubprocessOutputMode, reader: TensorlakeOutputReader | undefined, path: string): void => {
      if (!hasSpillMode(mode)) return
      // A spill mode is a collect mode, so construction always created its reader.
      const size = (reader as TensorlakeOutputReader).size
      if (this.outputDrainExpired || size <= mode.maxBytes || size > mode.spill.maxBytes) {
        removals.push(sandbox.deleteFile(path).catch((_adapterPrivateSpillRemovalFailure: unknown) => {
          // The command outcome is authoritative; owner teardown bounds private residue.
        }))
      }
    }
    collect(this.spec.stdio.stdout, this.stdoutReader, this.paths.stdoutSpill)
    collect(this.spec.stdio.stderr, this.stderrReader, this.paths.stderrSpill)
    await Promise.all(removals)
  }

  private async removeFailedState(sandbox: Sandbox): Promise<void> {
    try {
      const result = await sandbox.run('rm', { args: ['-rf', '--', this.stateDir] })
      if (result.exitCode !== 0) {
        throw new Error(`subprocess-tensorlake: failed to remove private command state: ${result.stderr.trim()}`)
      }
    } catch (error: unknown) {
      if (!isRemoteMissing(error)) throw error
    }
  }
}
