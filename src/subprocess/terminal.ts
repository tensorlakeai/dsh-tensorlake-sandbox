/** Tensorlake PTY allocation and process-session ownership for the subprocess seam. */

import { Buffer } from 'node:buffer'
import { PassThrough } from 'node:stream'
import { posix } from 'node:path'
import { isRemoteMissing, runChecked } from '../runtime.ts'
import type { Pty, Sandbox } from '../runtime.ts'
import type {
  SubprocessOutcome,
  SubprocessTerminalForeground,
  SubprocessTerminalHandle,
  SubprocessTerminalSignal,
  SubprocessTerminalSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'
import type TensorlakeRuntime from '../runtime.ts'
import { environmentArguments, readAmbientEnvironment } from './environment.ts'
import { asError, delay, parsePositiveId, TerminalOperationGate } from './provider.ts'
import { sessionProcessGroups, SHELL_NAME, signalRemoteGroups } from './remote.ts'

/**
 * The PTY leader publishes its own pid — the daemon makes it the session
 * leader — then execs the exact requested argv under a rebuilt environment.
 * `exec` keeps the pid, so the published id addresses both the terminal
 * process and its POSIX session, and nothing writes to the terminal before
 * the requested program owns it.
 */
const TERMINAL_BOOTSTRAP = [
  'printf \'%s\' "$$" > "$1"',
  'dsh_envc=$2',
  'shift 2',
  'dsh_env=( "${@:1:dsh_envc}" )',
  'shift "$dsh_envc"',
  'exec env -i -- "${dsh_env[@]}" "$@"',
].join('\n')

async function awaitSessionEmpty(
  sandbox: Sandbox,
  sessionId: number,
  graceMs: number,
  pollMs: number,
  kill = false,
): Promise<number[]> {
  const deadline = Date.now() + graceMs
  for (;;) {
    const groups = await sessionProcessGroups(sandbox, sessionId)
    if (groups.length === 0) return groups
    if (kill) {
      await signalRemoteGroups(sandbox, groups, 'KILL')
      if (Date.now() >= deadline) return await sessionProcessGroups(sandbox, sessionId)
    } else if (Date.now() >= deadline) {
      return groups
    }
    await delay(Math.min(pollMs, Math.max(1, deadline - Date.now())))
  }
}

/** One Tensorlake PTY and all process groups in its remote process session. */
export class TensorlakeTerminalHandle implements SubprocessTerminalHandle {
  readonly done: Promise<SubprocessOutcome>

  private topLevelExited = false
  private readonly gate = new TerminalOperationGate('subprocess-tensorlake: terminal is terminating')
  private terminationSignal: NodeJS.Signals | null = null

  constructor(
    private readonly sandbox: Sandbox,
    private readonly pty: Pty,
    readonly output: PassThrough,
    /** Published terminal pid; also the id of the POSIX session it leads. */
    readonly pid: number,
    private readonly stateDir: string,
    private readonly graceMs: number,
    private readonly pollMs: number,
  ) {
    this.done = this.waitForExit()
  }

  /** @inheritdoc */
  write(data: string): Promise<void> {
    return this.gate.run(async () => {
      if (this.topLevelExited) throw new Error('terminal process has exited')
      await this.pty.sendInput(data)
    })
  }

  /** @inheritdoc */
  inspectForeground(): Promise<SubprocessTerminalForeground | undefined> {
    return this.gate.run(() => this.inspectForegroundOnce())
  }

  /** @inheritdoc */
  signalForeground(signal: SubprocessTerminalSignal): Promise<number> {
    return this.gate.run(async () => {
      const foreground = await this.inspectForegroundOnce()
      if (foreground === undefined) {
        throw new Error(`subprocess-tensorlake: cannot resolve foreground process group for terminal ${this.pid}`)
      }
      if (signal === 'SIGKILL' && foreground.processGroupId === this.pid) {
        throw new Error('refusing to SIGKILL the terminal shell; terminate the terminal session instead')
      }
      await runChecked(this.sandbox, 'bash', [
        '-c',
        `kill -${signal.slice(3)} -- "-$1"`,
        SHELL_NAME,
        String(foreground.processGroupId),
      ])
      return foreground.processGroupId
    })
  }

  /** @inheritdoc */
  terminate(): Promise<void> {
    return this.gate.close(() => this.closeOnce())
  }

  private async inspectForegroundOnce(): Promise<SubprocessTerminalForeground | undefined> {
    const result = await this.sandbox.run('bash', {
      args: ['-c', 'ps -o tpgid= -p "$1"', SHELL_NAME, String(this.pid)],
    })
    if (result.exitCode !== 0) {
      if (this.topLevelExited || result.stdout.trim().length === 0) return undefined
      throw new Error(`subprocess-tensorlake: cannot resolve foreground process group for terminal ${this.pid}`)
    }
    return {
      processGroupId: parsePositiveId(
        result.stdout,
        `subprocess-tensorlake: cannot resolve foreground process group for terminal ${this.pid}`,
      ),
      // The sandbox exposes process-table commands but not the /proc memory
      // access needed to prove a specific syscall is waiting on fd 0.
      inputWaiting: false,
    }
  }

  private async waitForExit(): Promise<SubprocessOutcome> {
    try {
      const code = await this.pty.wait()
      // The websocket transport reports a killed session as a negative code
      // instead of an exit status.
      if (code >= 0 && this.terminationSignal === null) return { exitCode: code, signal: null }
      if (code >= 0) return { exitCode: null, signal: this.terminationSignal }
      return { exitCode: null, signal: this.terminationSignal ?? 'SIGKILL' }
    } catch (error: unknown) {
      this.output.destroy(asError(error))
      throw error
    } finally {
      this.topLevelExited = true
      if (!this.output.destroyed) this.output.end()
    }
  }

  /** Wait for the terminal process to report its outcome, bounded by the cleanup grace. */
  private async awaitTopLevelExit(): Promise<void> {
    await Promise.race([this.done.catch(() => undefined), delay(this.graceMs)])
  }

  private async closeOnce(): Promise<void> {
    let groups = await sessionProcessGroups(this.sandbox, this.pid)
    if (groups.length > 0) {
      this.terminationSignal = 'SIGTERM'
      await signalRemoteGroups(this.sandbox, groups, 'TERM')
      groups = await awaitSessionEmpty(this.sandbox, this.pid, this.graceMs, this.pollMs)
    }
    if (groups.length === 0 && !this.topLevelExited) await this.awaitTopLevelExit()
    if (groups.length > 0 || !this.topLevelExited) {
      this.terminationSignal = 'SIGKILL'
      if (!this.topLevelExited) {
        try {
          await this.pty.kill()
        } catch (error: unknown) {
          if (!isRemoteMissing(error)) throw error
        }
      }
      groups = await awaitSessionEmpty(this.sandbox, this.pid, this.graceMs, this.pollMs, true)
      if (!this.topLevelExited) await this.awaitTopLevelExit()
    }
    if (groups.length > 0) {
      throw new Error(`subprocess-tensorlake: terminal cleanup failed; surviving process groups: ${groups.join(', ')}`)
    }
    if (!this.topLevelExited) {
      throw new Error(`subprocess-tensorlake: terminal cleanup failed; surviving pid: ${this.pid}`)
    }
    try {
      await this.pty.kill()
    } catch (error: unknown) {
      if (!isRemoteMissing(error)) throw error
    }
    try {
      await runChecked(this.sandbox, 'rm', ['-rf', '--', this.stateDir])
    } catch {
      // The terminal is quiescent; owner teardown bounds private residue.
    }
  }
}

/**
 * Allocate a Tensorlake PTY running the requested argv as its session leader
 * and return only after the leader has published its process id.
 * @param runtime - Shared Tensorlake sandbox owner.
 * @param spec - Fully specified terminal-process request.
 * @param stateDir - Private remote directory for one startup transaction.
 * @param pollMs - Remote publication/liveness poll cadence.
 * @returns The live subprocess terminal handle.
 */
export async function spawnTensorlakeTerminal(
  runtime: TensorlakeRuntime,
  spec: SubprocessTerminalSpawnSpec,
  stateDir: string,
  pollMs: number,
): Promise<TensorlakeTerminalHandle> {
  const sandbox = await runtime.getSandbox()
  spec.signal?.throwIfAborted()
  const pidFile = posix.join(stateDir, 'pid')
  const output = new PassThrough()
  let pty: Pty | undefined
  let stateDirectoryCreated = false
  try {
    const ambient = await readAmbientEnvironment(sandbox)
    const envArgs = environmentArguments(ambient, spec.env)
    spec.signal?.throwIfAborted()
    stateDirectoryCreated = true
    await runChecked(sandbox, 'bash', [
      '-c',
      'set -e\nmkdir -p -- "$1"\nchmod 700 -- "$1"\n: > "$2"\nchmod 600 -- "$2"',
      SHELL_NAME,
      stateDir,
      pidFile,
    ])
    spec.signal?.throwIfAborted()
    pty = await sandbox.createPty({
      command: 'bash',
      args: ['-c', TERMINAL_BOOTSTRAP, SHELL_NAME, pidFile, String(envArgs.length), ...envArgs, ...spec.argv],
      workingDir: spec.cwd,
      rows: spec.rows,
      cols: spec.cols,
      onData: (data) => {
        if (!output.destroyed) output.write(Buffer.from(data))
      },
    })
    const exited = pty.wait().then(() => true, () => true)
    const pid = await waitForPublishedPid(sandbox, pidFile, exited, pollMs, spec.signal)
    return new TensorlakeTerminalHandle(sandbox, pty, output, pid, stateDir, spec.graceMs, pollMs)
  } catch (error: unknown) {
    output.destroy()
    const failures: Error[] = []
    if (pty !== undefined) {
      try {
        await pty.kill()
      } catch (cleanupError: unknown) {
        if (!isRemoteMissing(cleanupError)) failures.push(asError(cleanupError))
      }
    }
    if (stateDirectoryCreated) {
      try {
        await runChecked(sandbox, 'rm', ['-rf', '--', stateDir])
      } catch (stateError: unknown) {
        if (!isRemoteMissing(stateError)) failures.push(asError(stateError))
      }
    }
    if (failures.length > 0) {
      throw new AggregateError(
        [asError(error), ...failures],
        'subprocess-tensorlake: terminal setup cleanup did not complete',
      )
    }
    throw error
  }
}

async function waitForPublishedPid(
  sandbox: Sandbox,
  pidFile: string,
  exited: Promise<boolean>,
  pollMs: number,
  signal: AbortSignal | undefined,
): Promise<number> {
  for (;;) {
    signal?.throwIfAborted()
    const raw = new TextDecoder().decode(await sandbox.readFile(pidFile)).trim()
    if (raw.length > 0) {
      const pid = parsePositiveId(raw, `subprocess-tensorlake: terminal published an invalid process id ${JSON.stringify(raw)}`)
      // A same-UID sandbox process can rewrite this file; refuse ids whose
      // negative form addresses every process (`kill -- -1`) or init's group.
      if (pid <= 1) throw new Error(`subprocess-tensorlake: unsafe published terminal process id ${pid}`)
      return pid
    }
    const settled = await Promise.race([
      exited,
      new Promise<false>((resolve) => { setTimeout(() => { resolve(false) }, pollMs) }),
    ])
    if (settled) throw new Error('subprocess-tensorlake: terminal exited before publishing its process id')
  }
}
