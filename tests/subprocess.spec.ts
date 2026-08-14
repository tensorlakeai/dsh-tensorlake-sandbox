import { Buffer } from 'node:buffer'
import { once } from 'node:events'
import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, vi } from 'vitest'
import {
  ProcessStatus,
  RemoteAPIError,
  SandboxNotFoundError,
  StdinMode,
  type CommandResult,
  type OutputEvent,
  type ProcessInfo,
  type RunOptions,
  type Sandbox,
  type StartProcessOptions,
} from '../src/runtime.ts'
import type TensorlakeRuntime from '../src/runtime.ts'
import type { SubprocessSpawnSpec } from '@deepseek-ai/dsh-subprocess'
import TensorlakeSubprocessRuntime from '../src/subprocess/index.ts'
import { environmentArguments, readAmbientEnvironment } from '../src/subprocess/environment.ts'
import { OUTPUT_COMPLETE_MARKER, TensorlakeOutputDecoder, TensorlakeOutputReader } from '../src/subprocess/output.ts'
import { TensorlakeSubprocessHandle } from '../src/subprocess/process.ts'
import { asError, MAX_TIMER_DELAY_MS, parsePositiveId, waitTick } from '../src/subprocess/provider.ts'
import { groupAlive, sessionProcessGroups, signalRemoteGroups } from '../src/subprocess/remote.ts'

const AMBIENT = 'PATH=/ambient/bin\0KEEP=safe\0UNICODE=你好\0NPM_TOKEN=secret\0DSH_STALE=old\0BROKEN\0=bad\0'

/** Mirror of the remote `node -e` encoder: whole 3-byte quanta, then the marker. */
class RemoteEncoder {
  private carry = Buffer.alloc(0)

  write(data: Buffer): string[] {
    const combined = Buffer.concat([this.carry, data])
    const usable = combined.length - (combined.length % 3)
    this.carry = Buffer.from(combined.subarray(usable))
    return usable > 0 ? [combined.subarray(0, usable).toString('base64')] : []
  }

  finish(): string[] {
    const lines = this.carry.length > 0 ? [this.carry.toString('base64')] : []
    this.carry = Buffer.alloc(0)
    return [...lines, OUTPUT_COMPLETE_MARKER]
  }
}

/** Scriptable `followOutput` async iterable with a settled-consumer probe. */
class OutputChannel {
  finished = false
  private readonly queue: OutputEvent[] = []
  private wake: (() => void) | undefined
  private parked = false
  private ended = false
  private failure: unknown

  push(line: string, stream = 'stdout'): void {
    this.pushEvent({ line, timestamp: new Date(), stream })
  }

  /** Deliver one raw event, including one carrying no stream tag at all. */
  pushEvent(event: OutputEvent): void {
    this.queue.push(event)
    this.release()
  }

  end(): void {
    this.ended = true
    this.release()
  }

  fail(error: unknown): void {
    this.failure = error
    this.release()
  }

  /** Resolve once the follower has consumed everything, or after the bound. */
  async settle(boundMs = 60): Promise<boolean> {
    const deadline = Date.now() + boundMs
    for (;;) {
      if (this.finished || (this.parked && this.queue.length === 0)) return true
      if (Date.now() >= deadline) return false
      await new Promise(resolve => setTimeout(resolve, 1))
    }
  }

  async *iterate(signal: AbortSignal | undefined): AsyncIterable<OutputEvent> {
    try {
      for (;;) {
        const event = this.queue.shift()
        if (event !== undefined) {
          yield event
          continue
        }
        if (this.failure !== undefined) throw this.failure
        if (this.ended) return
        if (signal?.aborted === true) throw new Error('follow aborted')
        await new Promise<void>((resolve) => {
          const wake = (): void => { resolve() }
          this.parked = true
          this.wake = wake
          signal?.addEventListener('abort', wake, { once: true })
        })
      }
    } finally {
      this.finished = true
    }
  }

  private release(): void {
    this.parked = false
    const wake = this.wake
    this.wake = undefined
    wake?.()
  }
}

class FakeSandbox {
  readonly output = new OutputChannel()
  readonly encoders = { stdout: new RemoteEncoder(), stderr: new RemoteEncoder() }
  readonly runs: Array<{ command: string; args: string[]; options: RunOptions }> = []
  readonly signals: string[] = []
  readonly starts: Array<{ command: string; options: StartProcessOptions }> = []
  readonly prepared: string[][] = []
  readonly stdinData: string[] = []
  readonly removed: string[] = []
  readonly deleted: string[] = []
  readonly killed: number[] = []
  readonly signalErrors: unknown[] = []
  stdinCloses = 0
  statusReads = 0
  groupProbes = 0
  pid = 4242
  ambient = AMBIENT
  ambientWire: string | undefined
  ambientExit = 0
  ambientError: unknown
  prepareExit = 0
  startError: unknown
  status = ''
  readonly statusQueue: string[] = []
  statusError: unknown
  processStatus: ProcessStatus = ProcessStatus.RUNNING
  getProcessError: unknown
  killProcessError: unknown
  writeStdinError: unknown
  closeStdinError: unknown
  deleteFileError: unknown
  removeExit = 0
  removeError: unknown
  groupLive = true
  trapsTerm = false
  survivesKill = false
  probeExit = 0
  probeError: unknown
  signalExit = 0
  signalError: unknown
  sessionGroups: number[] = []
  sessionStdout: string | undefined
  sessionExit = 0
  sessionError: unknown
  executableProbeExit = 0
  lookupExit = 0
  lookupStdout = '/usr/bin/node\n'
  lookupError: unknown
  afterPrepare: (() => void) | undefined
  readFileGate: Promise<undefined> | undefined
  beforeStatusRead: (() => void) | undefined
  afterStatusRead: (() => void) | undefined
  beforeProbe: (() => void) | undefined
  afterProbe: (() => void) | undefined
  private runGate: Promise<undefined> | undefined
  private openRun: (() => void) | undefined
  private signalGate: Promise<undefined> | undefined
  private openSignal: (() => void) | undefined

  /** Hold the ambient-environment control command until {@link releaseRun}. */
  deferRun(): void {
    const gate = Promise.withResolvers<undefined>()
    this.runGate = gate.promise
    this.openRun = () => { gate.resolve(undefined) }
  }

  releaseRun(): void {
    this.openRun?.()
  }

  /** Hold every process-group signal until {@link releaseSignals}. */
  deferSignals(): void {
    const gate = Promise.withResolvers<undefined>()
    this.signalGate = gate.promise
    this.openSignal = () => { gate.resolve(undefined) }
  }

  releaseSignals(): void {
    this.openSignal?.()
  }

  /** The wrapper is the group leader, so a dead group is an exited daemon process. */
  private clearGroup(): void {
    this.groupLive = false
    this.sessionGroups = []
    this.processStatus = ProcessStatus.EXITED
  }

  /** Deliver one stream's encoded text, optionally split into arbitrary pieces. */
  async emit(stream: 'stdout' | 'stderr', text: string, pieceSize = 0): Promise<void> {
    await this.deliver(stream, this.encoders[stream].write(Buffer.from(text, 'utf8')), pieceSize)
  }

  async complete(stream: 'stdout' | 'stderr', pieceSize = 0): Promise<void> {
    await this.deliver(stream, this.encoders[stream].finish(), pieceSize)
  }

  async completeBoth(pieceSize = 0): Promise<void> {
    await this.complete('stdout', pieceSize)
    await this.complete('stderr', pieceSize)
  }

  private async deliver(stream: string, lines: string[], pieceSize: number): Promise<void> {
    for (const line of lines) {
      if (pieceSize <= 0) {
        this.output.push(line, stream)
        continue
      }
      for (let index = 0; index < line.length; index += pieceSize) {
        this.output.push(line.slice(index, index + pieceSize), stream)
      }
    }
    await this.output.settle()
  }

  readonly sandbox = {
    run: async (command: string, options: RunOptions = {}): Promise<CommandResult> => {
      const args = options.args ?? []
      this.runs.push({ command, args, options })
      if (command === 'rm') {
        this.removed.push(args.at(-1) ?? '')
        if (this.removeError !== undefined) throw this.removeError
        return { exitCode: this.removeExit, stdout: '', stderr: 'rm failed' }
      }
      const script = args[1] ?? ''
      if (script.includes('env -0 | base64')) {
        await this.runGate
        if (this.ambientError !== undefined) throw this.ambientError
        return {
          exitCode: this.ambientExit,
          stdout: this.ambientWire ?? Buffer.from(this.ambient, 'utf8').toString('base64'),
          stderr: 'ambient failed',
        }
      }
      if (script.includes('mkfifo')) {
        this.prepared.push(args.slice(3))
        this.afterPrepare?.()
        return { exitCode: this.prepareExit, stdout: '', stderr: 'prepare failed' }
      }
      if (script.startsWith('kill -')) {
        const signal = (script.split(' ')[1] ?? '').slice(1)
        this.signals.push(`${signal} ${args.slice(3).join(' ')}`)
        await this.signalGate
        const queued = this.signalErrors.shift()
        const failure = queued ?? this.signalError
        if (failure !== undefined) throw failure
        if (this.signalExit === 0 && (signal === 'TERM' ? !this.trapsTerm : !this.survivesKill)) {
          this.clearGroup()
        }
        return { exitCode: this.signalExit, stdout: '', stderr: 'kill failed' }
      }
      if (script.includes('ps -eo pgid=')) {
        this.groupProbes += 1
        this.beforeProbe?.()
        if (this.probeError !== undefined) throw this.probeError
        const stdout = this.groupLive ? 'live\n' : '\n'
        this.afterProbe?.()
        return { exitCode: this.probeExit, stdout, stderr: '' }
      }
      if (script.includes('ps -eo sid=')) {
        if (this.sessionError !== undefined) throw this.sessionError
        return {
          exitCode: this.sessionExit,
          stdout: this.sessionStdout ?? this.sessionGroups.map(group => `${group}\n`).join(''),
          stderr: '',
        }
      }
      if (script.includes('test -f')) {
        return { exitCode: this.executableProbeExit, stdout: '', stderr: '' }
      }
      if (script.includes('command -v')) {
        if (this.lookupError !== undefined) throw this.lookupError
        return { exitCode: this.lookupExit, stdout: this.lookupStdout, stderr: 'lookup failed' }
      }
      return { exitCode: 0, stdout: '', stderr: '' }
    },
    startProcess: async (command: string, options: StartProcessOptions = {}): Promise<ProcessInfo> => {
      this.starts.push({ command, options })
      if (this.startError !== undefined) throw this.startError
      return {
        pid: this.pid,
        status: ProcessStatus.RUNNING,
        stdinWritable: true,
        command,
        args: options.args ?? [],
        startedAt: new Date(),
      }
    },
    followOutput: (_pid: number, options?: { signal?: AbortSignal }): AsyncIterable<OutputEvent> =>
      this.output.iterate(options?.signal),
    getProcess: async (pid: number): Promise<ProcessInfo> => {
      if (this.getProcessError !== undefined) throw this.getProcessError
      return {
        pid,
        status: this.processStatus,
        stdinWritable: true,
        command: 'bash',
        args: [],
        startedAt: new Date(),
      }
    },
    readFile: async (_path: string): Promise<Uint8Array> => {
      this.beforeStatusRead?.()
      const gate = this.readFileGate
      this.readFileGate = undefined
      await gate
      this.statusReads += 1
      if (this.statusError !== undefined) {
        const failure = this.statusError
        this.statusError = undefined
        throw failure
      }
      const raw = this.statusQueue.length > 0 ? this.statusQueue.shift() ?? '' : this.status
      this.afterStatusRead?.()
      return new TextEncoder().encode(raw)
    },
    writeStdin: async (_pid: number, data: Uint8Array): Promise<void> => {
      if (this.writeStdinError !== undefined) throw this.writeStdinError
      this.stdinData.push(Buffer.from(data).toString('utf8'))
    },
    closeStdin: async (_pid: number): Promise<void> => {
      if (this.closeStdinError !== undefined) throw this.closeStdinError
      this.stdinCloses += 1
    },
    killProcess: async (pid: number): Promise<void> => {
      this.killed.push(pid)
      if (this.killProcessError !== undefined) throw this.killProcessError
      if (!this.survivesKill) this.clearGroup()
    },
    deleteFile: async (path: string): Promise<void> => {
      this.deleted.push(path)
      if (this.deleteFileError !== undefined) throw this.deleteFileError
    },
  } as unknown as Sandbox
}

function runtime(
  fake: FakeSandbox,
  getSandbox: () => Promise<Sandbox> = async () => fake.sandbox,
): TensorlakeRuntime {
  return {
    cwd: '/workspace',
    runtimeRoot: '/workspace/.dsh-tensorlake',
    getSandbox,
  } as unknown as TensorlakeRuntime
}

function spec(overrides: Partial<SubprocessSpawnSpec> = {}): SubprocessSpawnSpec {
  return {
    argv: ['tool', 'run'],
    cwd: '/workspace',
    stdio: {
      stdin: 'ignore',
      stdout: { maxBytes: 4, spill: { maxBytes: 16 } },
      stderr: { maxBytes: 4 },
    },
    graceMs: 5,
    ...overrides,
  }
}

/** Construct the handle under test with a short poll cadence. */
function testHandle(
  provided: TensorlakeRuntime,
  request: SubprocessSpawnSpec,
  stateDir: string,
  pollMs = 1,
): TensorlakeSubprocessHandle {
  return new TensorlakeSubprocessHandle(provided, request, stateDir, pollMs)
}

async function flush(): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, 0))
}

/** Wait until the remote command has started, so output and status can be scripted. */
async function started(fake: FakeSandbox): Promise<void> {
  await vi.waitFor(() => { expect(fake.starts).toHaveLength(1) })
  await flush()
}

function wrapperArguments(fake: FakeSandbox): {
  script: string
  paths: string[]
  encoder: string
  env: string[]
  argv: string[]
} {
  const args = fake.starts[0]?.options.args ?? []
  const count = Number(args[9])
  return {
    script: args[1] ?? '',
    paths: args.slice(3, 8),
    encoder: args[8] ?? '',
    env: args.slice(10, 10 + count),
    argv: args.slice(10 + count),
  }
}

describe('Tensorlake output transport', () => {
  it('decodes quanta across arbitrary transport splits and rejects malformed framing', () => {
    const decoder = new TensorlakeOutputDecoder()
    expect(decoder.push('')).toEqual(Buffer.alloc(0))
    expect(decoder.push('5')).toEqual(Buffer.alloc(0))
    expect(decoder.push('L2')).toEqual(Buffer.alloc(0))
    expect(decoder.push('g').toString('utf8')).toBe('你')
    expect(decoder.push('YWJj').toString('utf8')).toBe('abc')
    expect(decoder.isComplete).toBe(false)
    expect(decoder.push(OUTPUT_COMPLETE_MARKER.slice(0, 6))).toEqual(Buffer.alloc(0))
    expect(decoder.push(OUTPUT_COMPLETE_MARKER.slice(6))).toEqual(Buffer.alloc(0))
    expect(decoder.isComplete).toBe(true)
    decoder.finish()
    expect(() => decoder.push('YQ==')).toThrow('continued after completion')

    const trailing = new TensorlakeOutputDecoder()
    expect(trailing.push(`YQ==${OUTPUT_COMPLETE_MARKER}`).toString('utf8')).toBe('a')
    expect(() => new TensorlakeOutputDecoder().push('!not-the-marker')).toThrow('invalid output transport marker')
    expect(() => new TensorlakeOutputDecoder().push(`YQ${OUTPUT_COMPLETE_MARKER}`)).toThrow('invalid base64')
    expect(() => new TensorlakeOutputDecoder().push('%%%%')).toThrow('invalid base64')
    expect(() => new TensorlakeOutputDecoder().push('AB==')).toThrow('invalid base64')

    const truncated = new TensorlakeOutputDecoder()
    truncated.push('YQ')
    expect(() => { truncated.finish() }).toThrow('truncated base64')
    expect(() => { new TensorlakeOutputDecoder().finish() }).toThrow('incomplete output transport')
    const interrupted = new TensorlakeOutputDecoder()
    interrupted.push('YQ')
    expect(() => { interrupted.finish(false) }).not.toThrow()
  })

  it('keeps a byte-exact tail with independent whole-stream cursors', () => {
    const reader = new TensorlakeOutputReader(4, 10, '/remote/spill')
    reader.push(Buffer.alloc(0))
    reader.push(Buffer.from('ab'))
    reader.push(Buffer.from('cdef'))
    expect(reader.size).toBe(6)
    expect(reader.readFrom(0)).toEqual({ text: 'cdef', nextOffset: 6, lossy: true, spillPath: '/remote/spill' })
    expect(reader.readFrom(2)).toEqual({ text: 'cdef', nextOffset: 6, lossy: false })
    expect(reader.readFrom(5)).toEqual({ text: 'f', nextOffset: 6, lossy: false })
    expect(reader.readFrom(99)).toEqual({ text: '', nextOffset: 6, lossy: false })
    reader.invalidateSpill()
    expect(reader.readFrom(0)).toEqual({ text: 'cdef', nextOffset: 6, lossy: true })

    const withoutSpill = new TensorlakeOutputReader(2, undefined, '/unused')
    withoutSpill.push(Buffer.from('abcd'))
    expect(withoutSpill.readFrom(0)).toEqual({ text: 'cd', nextOffset: 4, lossy: true })
    const overCap = new TensorlakeOutputReader(2, 3, '/too-small')
    overCap.push(Buffer.from('abcd'))
    expect(overCap.readFrom(0)).toEqual({ text: 'cd', nextOffset: 4, lossy: true })
  })
})

describe('Tensorlake remote control helpers', () => {
  it('bounds poll ticks, parses ids, and normalizes rejections', async () => {
    await expect(waitTick(1, AbortSignal.abort())).resolves.toBe(false)
    await expect(waitTick(1)).resolves.toBe(true)
    const controller = new AbortController()
    const pending = waitTick(10_000, controller.signal)
    controller.abort()
    await expect(pending).resolves.toBe(false)

    expect(parsePositiveId(' 42\n', 'bad id')).toBe(42)
    expect(() => parsePositiveId('0', 'bad id')).toThrow('bad id')
    expect(() => parsePositiveId('nine', 'bad id')).toThrow('bad id')
    expect(() => parsePositiveId('99999999999999999999', 'bad id')).toThrow('bad id')
    const error = new Error('kept')
    expect(asError(error)).toBe(error)
    expect(asError('stringified').message).toBe('stringified')
  })

  it('tolerates already-gone groups and a disappeared sandbox when signalling', async () => {
    const fake = new FakeSandbox()
    await signalRemoteGroups(fake.sandbox, [7, 8], 'TERM')
    expect(fake.signals).toEqual(['TERM -7 -8'])

    fake.signalExit = 1
    await expect(signalRemoteGroups(fake.sandbox, [7], 'KILL')).resolves.toBeUndefined()
    fake.signalExit = 0
    fake.signalError = new SandboxNotFoundError('sandbox-gone')
    await expect(signalRemoteGroups(fake.sandbox, [7], 'KILL')).resolves.toBeUndefined()
    fake.signalError = new RemoteAPIError(404, 'missing')
    await expect(signalRemoteGroups(fake.sandbox, [7], 'KILL')).resolves.toBeUndefined()
    fake.signalError = new Error('signal transport failed')
    await expect(signalRemoteGroups(fake.sandbox, [7], 'KILL')).rejects.toThrow('signal transport failed')
  })

  it('reads group and session liveness through tolerant process-table probes', async () => {
    const fake = new FakeSandbox()
    await expect(groupAlive(fake.sandbox, 42)).resolves.toBe(true)
    fake.groupLive = false
    await expect(groupAlive(fake.sandbox, 42)).resolves.toBe(false)
    fake.groupLive = true
    fake.probeExit = 1
    await expect(groupAlive(fake.sandbox, 42)).resolves.toBe(false)
    fake.probeExit = 0
    fake.probeError = new SandboxNotFoundError('sandbox-gone')
    await expect(groupAlive(fake.sandbox, 42)).resolves.toBe(false)
    fake.probeError = new Error('probe transport failed')
    await expect(groupAlive(fake.sandbox, 42)).rejects.toThrow('probe transport failed')

    await expect(sessionProcessGroups(fake.sandbox, 9)).resolves.toEqual([])
    fake.sessionGroups = [11, 11, 12]
    await expect(sessionProcessGroups(fake.sandbox, 9)).resolves.toEqual([11, 12])
    fake.sessionStdout = 'not-a-group\n'
    await expect(sessionProcessGroups(fake.sandbox, 9)).rejects.toThrow('invalid process group')
    fake.sessionStdout = '1\n'
    await expect(sessionProcessGroups(fake.sandbox, 9)).rejects.toThrow('unsafe process group 1')
    fake.sessionStdout = undefined
    fake.sessionExit = 1
    await expect(sessionProcessGroups(fake.sandbox, 9)).resolves.toEqual([])
    fake.sessionExit = 0
    fake.sessionError = new SandboxNotFoundError('sandbox-gone')
    await expect(sessionProcessGroups(fake.sandbox, 9)).resolves.toEqual([])
    fake.sessionError = new Error('session transport failed')
    await expect(sessionProcessGroups(fake.sandbox, 9)).rejects.toThrow('session transport failed')
  })
})

describe('Tensorlake remote environment', () => {
  it('transports the ambient environment and scrubs private and credential names', async () => {
    const fake = new FakeSandbox()
    const ambient = await readAmbientEnvironment(fake.sandbox)
    expect([...ambient]).toEqual([
      ['PATH', '/ambient/bin'],
      ['KEEP', 'safe'],
      ['UNICODE', '你好'],
      ['NPM_TOKEN', 'secret'],
      ['DSH_STALE', 'old'],
    ])
    expect(environmentArguments(ambient, undefined)).toEqual([
      'PATH=/ambient/bin',
      'KEEP=safe',
      'UNICODE=你好',
    ])
    expect(environmentArguments(ambient, {
      PATH: '/bin',
      DEEPSEEK_API_KEY: 'explicit',
      DSH_MODE: 'test',
      KEEP: undefined,
    })).toEqual([
      'PATH=/bin',
      'UNICODE=你好',
      'DEEPSEEK_API_KEY=explicit',
      'DSH_MODE=test',
    ])
    for (const invalid of [{ '': 'x' }, { 'BAD=NAME': 'x' }, { 'BAD\0NAME': 'x' }, { BAD: 'x\0y' }]) {
      expect(() => environmentArguments(ambient, invalid)).toThrow('environment entries require')
    }
  })

  it('rejects an unreadable ambient environment transport', async () => {
    const nonzero = new FakeSandbox()
    nonzero.ambientExit = 3
    await expect(readAmbientEnvironment(nonzero.sandbox)).rejects.toThrow('bash exited 3')

    const malformed = new FakeSandbox()
    malformed.ambientWire = '%%%'
    await expect(readAmbientEnvironment(malformed.sandbox)).rejects.toThrow('invalid base64')

    const binary = new FakeSandbox()
    binary.ambientWire = Buffer.from([0xff, 0xfe]).toString('base64')
    await expect(readAmbientEnvironment(binary.sandbox)).rejects.toThrow('not valid UTF-8')
  })
})

describe('TensorlakeSubprocessHandle', () => {
  it('starts remotely, rebuilds the environment, and collects bounded output', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({
      argv: ['tool', 'argument with spaces'],
      stdio: {
        stdin: 'pipe',
        stdout: { maxBytes: 4, spill: { maxBytes: 32 } },
        stderr: 'pipe',
      },
      env: { PATH: '/bin', DEEPSEEK_API_KEY: 'explicit', DSH_MODE: 'test', KEEP: undefined },
    }), '/workspace/.dsh-tensorlake/processes/one')
    expect(handle.pid).toBe(-1)
    handle.stdin!.write('hello')
    handle.stdin!.end()
    await started(fake)
    expect(handle.pid).toBe(4242)

    expect(fake.prepared[0]).toEqual([
      '/workspace/.dsh-tensorlake/processes/one',
      '/workspace/.dsh-tensorlake/processes/one/stdout.pipe',
      '/workspace/.dsh-tensorlake/processes/one/stderr.pipe',
      '/workspace/.dsh-tensorlake/processes/one/exit-code',
      '/workspace/.dsh-tensorlake/processes/one/stdout.log',
    ])
    const wrapper = wrapperArguments(fake)
    expect(fake.starts[0]?.command).toBe('bash')
    expect(fake.starts[0]?.options).toMatchObject({
      workingDir: '/workspace',
      stdinMode: StdinMode.PIPE,
      stdoutMode: 'capture',
      stderrMode: 'capture',
    })
    expect(wrapper.paths).toEqual([
      '/workspace/.dsh-tensorlake/processes/one/exit-code',
      '/workspace/.dsh-tensorlake/processes/one/stdout.pipe',
      '/workspace/.dsh-tensorlake/processes/one/stderr.pipe',
      '/workspace/.dsh-tensorlake/processes/one/stdout.log',
      '/workspace/.dsh-tensorlake/processes/one/stderr.log',
    ])
    expect(wrapper.encoder).toContain("toString('base64')")
    expect(wrapper.encoder).toContain(OUTPUT_COMPLETE_MARKER)
    expect(wrapper.argv).toEqual(['tool', 'argument with spaces'])
    expect(wrapper.env).toEqual([
      'PATH=/bin',
      'UNICODE=你好',
      'DEEPSEEK_API_KEY=explicit',
      'DSH_MODE=test',
    ])
    expect(wrapper.script).toContain('tee --output-error=warn-nopipe >(head -c 32 > "$dsh_out_spill")')
    expect(wrapper.script).toContain('{ node -e "$dsh_encoder"; } < "$dsh_err_pipe" >&2 &')
    expect(wrapper.script).toContain('env -i -- "${dsh_env[@]}" "$@" > "$dsh_out_pipe" 2> "$dsh_err_pipe"')
    expect(fake.stdinData).toEqual(['hello'])
    expect(fake.stdinCloses).toBe(1)

    const piped: Buffer[] = []
    handle.stderr!.on('data', (chunk: Buffer) => { piped.push(chunk) })
    await fake.emit('stdout', 'abcdef')
    await fake.emit('stderr', 'A你好B', 3)
    await fake.completeBoth(5)
    fake.status = '0\n'

    await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
    expect(Buffer.concat(piped).toString('utf8')).toBe('A你好B')
    expect(handle.collected.stdout!.readFrom(0)).toEqual({
      text: 'cdef',
      nextOffset: 6,
      lossy: true,
      spillPath: '/workspace/.dsh-tensorlake/processes/one/stdout.log',
    })
    expect(handle.collected.stderr).toBeUndefined()
    expect(fake.deleted).toEqual([])

    fake.groupLive = false
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('writes batch stdin, tolerates a closed input, and removes an undersized spill', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({
      stdio: {
        stdin: { data: 'batch' },
        stdout: { maxBytes: 8, spill: { maxBytes: 16 } },
        stderr: { maxBytes: 2, spill: { maxBytes: 3 } },
      },
    }), '/runtime/batch')
    await started(fake)
    expect(fake.stdinData).toEqual(['batch'])
    expect(fake.stdinCloses).toBe(1)
    expect(fake.starts[0]?.options.stdinMode).toBe(StdinMode.PIPE)
    expect(wrapperArguments(fake).script).toContain('head -c 3 > "$dsh_err_spill"')

    await fake.emit('stdout', 'abc')
    await fake.emit('stderr', 'abcdef')
    await fake.completeBoth()
    fake.status = '7\n'
    await expect(handle.done).resolves.toEqual({ exitCode: 7, signal: null })
    // stdout fits its in-memory cap and stderr overruns its spill cap: both spills go.
    expect(fake.deleted).toEqual([
      '/runtime/batch/stdout.log',
      '/runtime/batch/stderr.log',
    ])
    expect(handle.collected.stderr!.readFrom(0)).toEqual({ text: 'ef', nextOffset: 6, lossy: true })

    const closed = new FakeSandbox()
    closed.writeStdinError = new Error('process already closed its input')
    const tolerant = testHandle(runtime(closed), spec({
      stdio: { stdin: { data: 'ignored' }, stdout: { maxBytes: 4 }, stderr: { maxBytes: 4 } },
    }), '/runtime/batch-closed')
    await started(closed)
    await closed.completeBoth()
    closed.status = '0\n'
    await expect(tolerant.done).resolves.toEqual({ exitCode: 0, signal: null })
    expect(closed.deleted).toEqual([])
  })

  it('tolerates a failed spill removal', async () => {
    const fake = new FakeSandbox()
    fake.deleteFileError = new Error('spill already removed')
    const handle = testHandle(runtime(fake), spec({
      stdio: { stdin: 'ignore', stdout: { maxBytes: 8, spill: { maxBytes: 16 } }, stderr: 'inherit' },
    }), '/runtime/spill-removal')
    await started(fake)
    await fake.emit('stdout', 'ab')
    await fake.completeBoth()
    fake.status = '0\n'
    await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
    expect(fake.deleted).toEqual(['/runtime/spill-removal/stdout.log'])
  })

  it('surfaces deferred piped-stdin write and close failures as stream errors', async () => {
    const writeFake = new FakeSandbox()
    writeFake.writeStdinError = 'stdin rejected'
    const writeHandle = testHandle(runtime(writeFake), spec({
      stdio: { stdin: 'pipe', stdout: { maxBytes: 4 }, stderr: { maxBytes: 4 } },
    }), '/runtime/stdin-write-error')
    const writeError = once(writeHandle.stdin!, 'error')
    writeHandle.stdin!.write('input')
    await expect(writeError).resolves.toMatchObject([{ message: 'stdin rejected' }])
    await writeFake.completeBoth()
    writeFake.status = '0\n'
    await writeHandle.done

    const closeFake = new FakeSandbox()
    closeFake.closeStdinError = new Error('close rejected')
    const closeHandle = testHandle(runtime(closeFake), spec({
      stdio: { stdin: 'pipe', stdout: { maxBytes: 4 }, stderr: { maxBytes: 4 } },
    }), '/runtime/stdin-close-error')
    await started(closeFake)
    const closeError = once(closeHandle.stdin!, 'error')
    closeHandle.stdin!.end()
    await expect(closeError).resolves.toMatchObject([{ message: 'close rejected' }])
    await closeFake.completeBoth()
    closeFake.status = '0\n'
    await closeHandle.done
  })

  it('fails a piped stdin write once startup rejects', async () => {
    const fake = new FakeSandbox()
    fake.startError = new Error('start failed')
    const handle = testHandle(runtime(fake), spec({
      stdio: { stdin: 'pipe', stdout: { maxBytes: 4 }, stderr: { maxBytes: 4 } },
    }), '/runtime/stdin-start-failure')
    const failure = once(handle.stdin!, 'error')
    await expect(handle.done).rejects.toThrow('start failed')
    handle.stdin!.write('input')
    await expect(failure).resolves.toMatchObject([{ message: 'start failed' }])
  })

  it('writes inherited output to the harness streams', async () => {
    const fake = new FakeSandbox()
    const stdout: string[] = []
    const stderr: string[] = []
    const outSpy = vi.spyOn(process.stdout, 'write').mockImplementation(((chunk: Uint8Array) => {
      stdout.push(Buffer.from(chunk).toString('utf8'))
      return true
    }) as typeof process.stdout.write)
    const errSpy = vi.spyOn(process.stderr, 'write').mockImplementation(((chunk: Uint8Array) => {
      stderr.push(Buffer.from(chunk).toString('utf8'))
      return true
    }) as typeof process.stderr.write)
    try {
      const handle = testHandle(runtime(fake), spec({
        stdio: { stdin: 'ignore', stdout: 'inherit', stderr: 'inherit' },
      }), '/runtime/inherit')
      await started(fake)
      expect(fake.starts[0]?.options.stdinMode).toBe(StdinMode.CLOSED)
      await fake.emit('stdout', 'out')
      await fake.emit('stderr', 'err')
      await fake.completeBoth()
      fake.status = '0\n'
      await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
      expect(stdout.join('')).toBe('out')
      expect(stderr.join('')).toBe('err')
    } finally {
      outSpy.mockRestore()
      errSpy.mockRestore()
    }
  })

  it('applies pipe backpressure and releases it on drain, close, and termination', async () => {
    const drained = new FakeSandbox()
    const drainHandle = testHandle(runtime(drained), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/backpressure-drain')
    await started(drained)
    const drainWrite = vi.spyOn(drainHandle.stdout!, 'write').mockReturnValueOnce(false)
    const blocked = drained.emit('stdout', 'blocked')
    await flush()
    queueMicrotask(() => { drainHandle.stdout!.emit('drain') })
    await blocked
    drainWrite.mockRestore()
    await drained.completeBoth()
    drained.status = '0\n'
    await expect(drainHandle.done).resolves.toEqual({ exitCode: 0, signal: null })

    const closed = new FakeSandbox()
    const closeHandle = testHandle(runtime(closed), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/backpressure-close')
    await started(closed)
    const closeWrite = vi.spyOn(closeHandle.stdout!, 'write').mockReturnValueOnce(false)
    const discarded = closed.emit('stdout', 'discarded')
    await flush()
    closeHandle.stdout!.destroy()
    await discarded
    closeWrite.mockRestore()
    await closed.emit('stdout', 'after close')
    closed.status = '0\n'
    await expect(closeHandle.done).rejects.toThrow('subprocess output stream is closed')

    const terminated = new FakeSandbox()
    const terminatedHandle = testHandle(runtime(terminated), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/backpressure-termination')
    await started(terminated)
    const terminatedWrite = vi.spyOn(terminatedHandle.stdout!, 'write').mockReturnValue(false)
    terminated.output.push(Buffer.from('blocked').toString('base64'), 'stdout')
    terminated.output.push(Buffer.from('late').toString('base64'), 'stdout')
    await flush()
    terminatedHandle.terminate()
    await expect(terminatedHandle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    terminatedWrite.mockRestore()
  })

  it('breaks backpressure when a synchronous pipe write starts termination', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/backpressure-synchronous')
    await started(fake)
    const write = vi.spyOn(handle.stdout!, 'write').mockImplementationOnce(() => {
      handle.terminate()
      return false
    })
    await expect(fake.emit('stdout', 'blocked')).resolves.toBeUndefined()
    write.mockRestore()
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
  })

  it('reports a sink failure raised while the pipe is blocked', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/backpressure-sink-failure')
    await started(fake)
    handle.stdout!.on('error', () => {})
    const write = vi.spyOn(handle.stdout!, 'write').mockReturnValueOnce(false)
    const pending = fake.emit('stdout', 'blocked')
    await flush()
    handle.stdout!.emit('error', new Error('sink failed'))
    await pending
    write.mockRestore()
    fake.status = '0\n'
    await expect(handle.done).rejects.toThrow('sink failed')
  })

  it('rejects malformed transport frames and unknown stream tags', async () => {
    const malformed = new FakeSandbox()
    const malformedHandle = testHandle(runtime(malformed), spec(), '/runtime/malformed-output')
    await started(malformed)
    malformed.output.push('%%%%', 'stdout')
    await malformed.output.settle()
    malformed.status = '0\n'
    await expect(malformedHandle.done).rejects.toThrow('invalid base64 output transport')

    const unknown = new FakeSandbox()
    const unknownHandle = testHandle(runtime(unknown), spec(), '/runtime/unknown-stream')
    await started(unknown)
    unknown.output.pushEvent({ line: 'YQ==', timestamp: new Date() })
    await unknown.output.settle()
    unknown.status = '0\n'
    await expect(unknownHandle.done).rejects.toThrow('unknown stream undefined')

    const late = new FakeSandbox()
    const lateHandle = testHandle(runtime(late), spec(), '/runtime/late-output')
    await started(late)
    await late.complete('stdout')
    late.output.push('YQ==', 'stdout')
    await late.output.settle()
    late.status = '0\n'
    await expect(lateHandle.done).rejects.toThrow('continued after completion')

    const broken = new FakeSandbox()
    const brokenHandle = testHandle(runtime(broken), spec(), '/runtime/broken-transport')
    await started(broken)
    broken.output.fail(new Error('output transport failed'))
    await broken.output.settle()
    broken.status = '0\n'
    await expect(brokenHandle.done).rejects.toThrow('output transport failed')
  })

  it('bounds descendant-held draining and withholds the incomplete spill', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({ graceMs: 5 }), '/runtime/drain-bound')
    await started(fake)
    await fake.emit('stdout', 'leader-output')
    fake.status = '0\n'

    await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
    expect(handle.collected.stdout!.readFrom(0)).toEqual({ text: 'utpu', nextOffset: 12, lossy: true })
    expect(fake.deleted).toContain('/runtime/drain-bound/stdout.log')
    fake.groupLive = false
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('reports the requested signal over a status published during termination', async () => {
    const fake = new FakeSandbox()
    fake.trapsTerm = true
    const handle = testHandle(runtime(fake), spec({ graceMs: 100 }), '/runtime/signal-over-status')
    await started(fake)
    handle.terminate()
    fake.status = '7\n'
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    fake.groupLive = false
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('contains a failing pipe teardown while reporting the transport failure', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({
      stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 4 } },
    }), '/runtime/failing-teardown')
    await started(fake)
    const teardown = Promise.withResolvers<undefined>()
    vi.spyOn(handle.stdout!, 'destroy').mockImplementationOnce(() => {
      teardown.resolve(undefined)
      throw new Error('pipe teardown failed')
    })
    // Deliver the malformed frame and hold the status read until the failing
    // teardown has run, so the adapter contains that rejection in one turn.
    fake.beforeStatusRead = () => {
      fake.beforeStatusRead = undefined
      fake.output.push('%%%%', 'stdout')
      fake.status = '0\n'
      fake.readFileGate = teardown.promise
    }
    await expect(handle.done).rejects.toThrow('invalid base64 output transport')
  })

  it('returns false when the caller aborts as the group probe reports quiescence', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec(), '/runtime/abort-at-quiescence')
    await started(fake)
    const controller = new AbortController()
    fake.beforeProbe = () => {
      fake.beforeProbe = undefined
      fake.groupLive = false
      controller.abort()
    }
    await expect(handle.waitForExit(controller.signal)).resolves.toBe(false)
    await fake.completeBoth()
    fake.status = '0\n'
    await handle.done
  })

  it('rejects an exit status the remote wrapper cannot have published', async () => {
    const invalid: Array<[string, string]> = [['999\n', 'over'], ['abc\n', 'text'], ['99999999999999999999\n', 'huge']]
    for (const [raw, stateDir] of invalid) {
      const fake = new FakeSandbox()
      const handle = testHandle(runtime(fake), spec(), `/runtime/invalid-status-${stateDir}`)
      await started(fake)
      fake.status = raw
      await expect(handle.done).rejects.toThrow('published invalid exit code')
      fake.groupLive = false
      await expect(handle.waitForExit()).resolves.toBe(true)
    }
  })

  it('closes the exit-then-publish race before concluding wrapper silence', async () => {
    const fake = new FakeSandbox()
    fake.statusQueue.push('')
    fake.status = '5\n'
    fake.processStatus = ProcessStatus.EXITED
    const handle = testHandle(runtime(fake), spec(), '/runtime/publish-race')
    await started(fake)
    await fake.completeBoth()
    await expect(handle.done).resolves.toEqual({ exitCode: 5, signal: null })
    expect(fake.statusReads).toBeGreaterThanOrEqual(3)
  })

  it('rejects a wrapper that exited without publishing its exit status', async () => {
    const fake = new FakeSandbox()
    fake.processStatus = ProcessStatus.EXITED
    const handle = testHandle(runtime(fake), spec(), '/runtime/silent-wrapper')
    await expect(handle.done).rejects.toThrow('exited before publishing its exit status')

    const missing = new FakeSandbox()
    missing.getProcessError = new SandboxNotFoundError('sandbox-gone')
    const missingHandle = testHandle(runtime(missing), spec(), '/runtime/silent-missing')
    await expect(missingHandle.done).rejects.toThrow('exited before publishing its exit status')

    const failed = new FakeSandbox()
    failed.getProcessError = new Error('process lookup failed')
    const failedHandle = testHandle(runtime(failed), spec(), '/runtime/status-lookup-failure')
    await expect(failedHandle.done).rejects.toThrow('process lookup failed')
  })

  it('reports the requested signal when a terminated wrapper exits silently', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec({ graceMs: 1 }), '/runtime/terminated-silence')
    await started(fake)
    handle.terminate()
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    expect(fake.signals).toEqual(['TERM -4242'])
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('reports SIGKILL when cancellation outruns the termination transaction', async () => {
    const fake = new FakeSandbox()
    const reconnect = Promise.withResolvers<Sandbox>()
    let calls = 0
    const handle = testHandle(runtime(fake, async () => {
      calls += 1
      return calls === 1 ? fake.sandbox : await reconnect.promise
    }), spec(), '/runtime/cancel-before-signal')
    await started(fake)
    await fake.completeBoth()
    fake.processStatus = ProcessStatus.EXITED
    handle.terminate()

    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })
    reconnect.resolve(fake.sandbox)
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('rolls back an unsafe remote process id', async () => {
    for (const pid of [0, 1]) {
      const fake = new FakeSandbox()
      fake.pid = pid
      const handle = testHandle(runtime(fake), spec(), `/runtime/unsafe-pid-${pid}`)
      await expect(handle.done).rejects.toThrow(`Tensorlake returned unsafe process id ${pid}`)
      expect(fake.killed).toEqual([pid])
      expect(fake.removed).toEqual([`/runtime/unsafe-pid-${pid}`])
      await expect(handle.waitForExit()).resolves.toBe(true)
    }

    const reaped = new FakeSandbox()
    reaped.pid = 0
    reaped.killProcessError = new RemoteAPIError(400, 'process is not running')
    const reapedHandle = testHandle(runtime(reaped), spec(), '/runtime/unsafe-pid-reaped')
    await expect(reapedHandle.done).rejects.toThrow('unsafe process id 0')
    await expect(reapedHandle.waitForExit()).resolves.toBe(true)

    const retained = new FakeSandbox()
    retained.pid = 0
    retained.killProcessError = new Error('rollback kill failed')
    const retainedHandle = testHandle(runtime(retained), spec(), '/runtime/unsafe-pid-retained')
    await expect(retainedHandle.done).rejects.toThrow('rollback kill failed')
    await expect(retainedHandle.waitForExit()).resolves.toBe(true)
  })

  it('reports private state setup and removal failures', async () => {
    const setup = new FakeSandbox()
    setup.prepareExit = 1
    const setupHandle = testHandle(runtime(setup), spec(), '/runtime/prepare-failure')
    await expect(setupHandle.done).rejects.toThrow('private state setup exited 1: prepare failed')
    expect(setup.removed).toEqual(['/runtime/prepare-failure'])

    const removal = new FakeSandbox()
    removal.startError = new Error('start failed')
    removal.removeExit = 1
    const removalHandle = testHandle(runtime(removal), spec(), '/runtime/removal-failure')
    const failure = await removalHandle.done.catch((error: unknown) => error as AggregateError)
    expect(failure).toBeInstanceOf(AggregateError)
    expect((failure as AggregateError).message).toContain('private state cleanup failed')
    expect((failure as AggregateError).errors[1]).toMatchObject({
      message: 'subprocess-tensorlake: failed to remove private command state: rm failed',
    })

    const gone = new FakeSandbox()
    gone.startError = new Error('start failed after external cleanup')
    gone.removeError = new SandboxNotFoundError('sandbox-gone')
    const goneHandle = testHandle(runtime(gone), spec(), '/runtime/removal-missing')
    await expect(goneHandle.done).rejects.toThrow('start failed after external cleanup')

    const transport = new FakeSandbox()
    transport.startError = new Error('start failed before cleanup transport failed')
    transport.removeError = new Error('remove transport failed')
    const transportHandle = testHandle(runtime(transport), spec(), '/runtime/removal-transport')
    await expect(transportHandle.done).rejects.toThrow('private state cleanup failed')

    const unavailable = testHandle(
      runtime(new FakeSandbox(), async () => { throw new Error('sandbox unavailable') }),
      spec(),
      '/runtime/unavailable-start',
    )
    await expect(unavailable.done).rejects.toThrow('sandbox unavailable')
    await expect(unavailable.waitForExit()).resolves.toBe(true)
  })

  it('settles a spawn canceled during preparation as SIGTERM without remote residue', async () => {
    const fake = new FakeSandbox()
    fake.deferRun()
    const handle = testHandle(runtime(fake), spec(), '/runtime/canceled-preparation')
    await vi.waitFor(() => { expect(fake.runs).toHaveLength(1) })
    handle.terminate()
    fake.releaseRun()

    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    expect(fake.starts).toEqual([])
    expect(fake.removed).toEqual([])
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('cleans committed private state when preparation is canceled after the directory exists', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec(), '/runtime/canceled-after-prepare')
    // The adapter's first remote round-trip has not returned yet, so this hook
    // lands before the preparation it cancels.
    fake.afterPrepare = () => { handle.terminate() }
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    expect(fake.starts).toEqual([])
    expect(fake.removed).toEqual(['/runtime/canceled-after-prepare'])
  })

  it('terminates a live process group with TERM and makes quiescence permanent', async () => {
    const fake = new FakeSandbox()
    const handle = testHandle(runtime(fake), spec(), '/runtime/term')
    await started(fake)
    handle.terminate()
    handle.terminate()
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    await expect(handle.waitForExit()).resolves.toBe(true)
    expect(fake.signals).toEqual(['TERM -4242'])

    fake.groupLive = true
    handle.terminate()
    await flush()
    expect(fake.signals).toEqual(['TERM -4242'])
  })

  it('escalates a TERM-trapping group to KILL and surfaces a surviving group', async () => {
    const fake = new FakeSandbox()
    fake.trapsTerm = true
    fake.survivesKill = true
    fake.killProcessError = new Error('daemon kill failed')
    const handle = testHandle(runtime(fake), spec({ graceMs: 1 }), '/runtime/force')
    await started(fake)
    handle.terminate()
    await expect(handle.waitForExit()).rejects.toThrow('remained live after force termination')
    expect(fake.signals.slice(0, 2)).toEqual(['TERM -4242', 'KILL -4242'])
    expect(fake.killed).toContain(4242)
    await expect(handle.done).rejects.toThrow('process-group rollback did not reach quiescence')

    fake.survivesKill = false
    fake.killProcessError = undefined
    handle.terminate()
    await expect(handle.waitForExit()).resolves.toBe(true)
  })

  it('escalates when the graceful transport itself fails', async () => {
    const fake = new FakeSandbox()
    fake.signalErrors.push(new Error('TERM transport failed'))
    const handle = testHandle(runtime(fake), spec({ graceMs: 1 }), '/runtime/failed-term')
    await started(fake)
    handle.terminate()
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })
    expect(fake.signals).toEqual(['TERM -4242', 'KILL -4242'])
  })

  it('keeps proven quiescence after a concurrent termination transport fails', async () => {
    const fake = new FakeSandbox()
    fake.trapsTerm = true
    fake.survivesKill = true
    fake.killProcessError = new Error('daemon kill failed')
    fake.deferSignals()
    const handle = testHandle(runtime(fake), spec({ graceMs: 1 }), '/runtime/quiescent-race')
    await started(fake)
    handle.terminate()
    await vi.waitFor(() => { expect(fake.signals).toContain('TERM -4242') })
    fake.groupLive = false
    fake.processStatus = ProcessStatus.EXITED
    await expect(handle.waitForExit()).resolves.toBe(true)

    fake.probeError = new Error('post-quiescence probe failed')
    fake.releaseSignals()
    await expect(handle.done).resolves.toMatchObject({ exitCode: null })
    const signals = fake.signals.length
    handle.terminate()
    await expect(handle.waitForExit()).resolves.toBe(true)
    expect(fake.signals).toHaveLength(signals)
  })

  it('treats a disappeared sandbox as quiescent during termination', async () => {
    const fake = new FakeSandbox()
    let calls = 0
    const handle = testHandle(runtime(fake, async () => {
      calls += 1
      if (calls === 1) return fake.sandbox
      throw new SandboxNotFoundError('sandbox-gone')
    }), spec(), '/runtime/expired-termination')
    await started(fake)
    await fake.completeBoth()
    handle.terminate()
    await expect(handle.waitForExit()).resolves.toBe(true)
    fake.status = '0\n'
    await handle.done
  })

  it('propagates a termination readiness failure that is not a missing sandbox', async () => {
    const fake = new FakeSandbox()
    let calls = 0
    const handle = testHandle(runtime(fake, async () => {
      calls += 1
      if (calls === 1) return fake.sandbox
      throw new Error('connection unavailable')
    }), spec(), '/runtime/termination-readiness')
    await started(fake)
    handle.terminate()
    await expect(handle.waitForExit()).rejects.toThrow('connection unavailable')
    await expect(handle.done).rejects.toThrow('process-group rollback did not reach quiescence')
    expect(calls).toBeGreaterThan(1)
  })

  it('rolls back a published group before rejecting a monitoring failure', async () => {
    const fake = new FakeSandbox()
    fake.statusError = new Error('status transport failed')
    const handle = testHandle(runtime(fake), spec(), '/runtime/monitoring-failure')
    await expect(handle.done).rejects.toThrow('status transport failed')
    expect(fake.signals).toContain('TERM -4242')
    await expect(handle.waitForExit()).resolves.toBe(true)

    const retained = new FakeSandbox()
    retained.statusError = new Error('status transport failed')
    retained.trapsTerm = true
    retained.survivesKill = true
    retained.killProcessError = new Error('daemon kill failed')
    const retainedHandle = testHandle(runtime(retained), spec({ graceMs: 1 }), '/runtime/monitoring-rollback')
    await expect(retainedHandle.done).rejects.toThrow('process-group rollback did not reach quiescence')
    retained.survivesKill = false
    retained.killProcessError = undefined
    retainedHandle.terminate()
    await expect(retainedHandle.waitForExit()).resolves.toBe(true)
  })

  it('honors an abort signal before and after the remote command starts', async () => {
    const fake = new FakeSandbox()
    const preAborted = testHandle(runtime(fake), spec({ signal: AbortSignal.abort('stop') }), '/runtime/pre-aborted')
    await expect(preAborted.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })

    const live = new FakeSandbox()
    const controller = new AbortController()
    const liveHandle = testHandle(runtime(live), spec({ signal: controller.signal }), '/runtime/live-abort')
    await started(live)
    controller.abort('stop')
    await expect(liveHandle.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
  })

  it('bounds waitForExit across startup, polling, and sandbox loss', async () => {
    const fake = new FakeSandbox()
    fake.deferRun()
    const handle = testHandle(runtime(fake), spec(), '/runtime/wait-bounds')
    const beforeStart = new AbortController()
    const pending = handle.waitForExit(beforeStart.signal)
    beforeStart.abort()
    await expect(pending).resolves.toBe(false)
    await expect(handle.waitForExit(AbortSignal.abort())).resolves.toBe(false)
    fake.releaseRun()
    await started(fake)

    const duringProbe = new AbortController()
    fake.beforeProbe = () => {
      fake.beforeProbe = undefined
      duringProbe.abort()
    }
    await expect(handle.waitForExit(duringProbe.signal)).resolves.toBe(false)

    const duringTick = new AbortController()
    fake.afterProbe = () => {
      fake.afterProbe = undefined
      setTimeout(() => { duringTick.abort() }, 0)
    }
    await expect(handle.waitForExit(duringTick.signal)).resolves.toBe(false)

    fake.probeError = new Error('probe failed')
    await expect(handle.waitForExit()).rejects.toThrow('probe failed')
    fake.probeError = undefined

    setTimeout(() => { fake.groupLive = false }, 2)
    await expect(handle.waitForExit(new AbortController().signal)).resolves.toBe(true)

    await fake.completeBoth()
    fake.status = '0\n'
    await handle.done
  })

  it('treats startup failure and sandbox loss as no live tree', async () => {
    const failed = new FakeSandbox()
    failed.startError = new Error('start failed')
    const failedHandle = testHandle(runtime(failed), spec(), '/runtime/start-failure')
    await expect(failedHandle.done).rejects.toThrow('start failed')
    expect(failedHandle.pid).toBe(-1)
    await expect(failedHandle.waitForExit()).resolves.toBe(true)
    failedHandle.terminate()

    const expired = new FakeSandbox()
    let calls = 0
    const expiredHandle = testHandle(runtime(expired, async () => {
      calls += 1
      if (calls === 1) return expired.sandbox
      throw new SandboxNotFoundError('sandbox-gone')
    }), spec(), '/runtime/expired-liveness')
    await started(expired)
    await expect(expiredHandle.waitForExit()).resolves.toBe(true)
    await expired.completeBoth()
    expired.status = '0\n'
    await expiredHandle.done

    const reconnecting = new FakeSandbox()
    const reconnect = Promise.withResolvers<Sandbox>()
    let attempts = 0
    const reconnectingHandle = testHandle(runtime(reconnecting, async () => {
      attempts += 1
      return attempts === 1 ? reconnecting.sandbox : await reconnect.promise
    }), spec(), '/runtime/reconnect-abort')
    await started(reconnecting)
    const controller = new AbortController()
    const waiting = reconnectingHandle.waitForExit(controller.signal)
    await flush()
    controller.abort()
    reconnect.reject(new Error('connection unavailable'))
    await expect(waiting).resolves.toBe(false)
    await reconnecting.completeBoth()
    reconnecting.status = '0\n'
    await reconnectingHandle.done
  })
})

describe('TensorlakeSubprocessRuntime', () => {
  async function service(
    fake = new FakeSandbox(),
    provided: TensorlakeRuntime = runtime(fake),
  ): Promise<{ ctx: Context; fiber: Awaited<ReturnType<Context['plugin']>> }> {
    const ctx = new Context()
    ctx.provide('tensorlake', provided as never)
    const fiber = await ctx.plugin(TensorlakeSubprocessRuntime, { pollMs: 1 })
    return { ctx, fiber }
  }

  it('validates synchronous spawn preconditions', async () => {
    const { ctx } = await service()
    expect(() => ctx.subprocess.spawn(spec({ argv: [] }))).toThrow('non-empty program name')
    expect(() => ctx.subprocess.spawn(spec({ argv: [''] }))).toThrow('non-empty program name')
    expect(() => ctx.subprocess.spawn(spec({ signal: AbortSignal.abort('stop') }))).toThrow('aborted before spawn')
    for (const graceMs of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, MAX_TIMER_DELAY_MS + 1]) {
      expect(() => ctx.subprocess.spawn(spec({ graceMs }))).toThrow('graceMs must be a positive finite number')
    }
  })

  it('rejects a non-positive poll cadence at load', async () => {
    const ctx = new Context()
    ctx.provide('tensorlake', runtime(new FakeSandbox()) as never)
    await expect(ctx.plugin(TensorlakeSubprocessRuntime, { pollMs: 0 }))
      .rejects.toThrow('pollMs must be a positive safe integer')
    await expect(ctx.plugin(TensorlakeSubprocessRuntime, { pollMs: 1.5 }))
      .rejects.toThrow('pollMs must be a positive safe integer')
    const defaulted = await ctx.plugin(TensorlakeSubprocessRuntime)
    await defaulted.dispose()
  })

  it('resolves remote executables', async () => {
    const fake = new FakeSandbox()
    const { ctx } = await service(fake)
    await expect(ctx.subprocess.resolveExecutable('/bin/bash')).resolves.toBe('/bin/bash')
    await expect(ctx.subprocess.resolveExecutable('node', { PATH: '/custom/bin' }, new AbortController().signal))
      .resolves.toBe('/usr/bin/node')
    expect(fake.runs.at(-1)?.options).toMatchObject({ workingDir: '/workspace', env: { PATH: '/custom/bin' } })
    fake.lookupStdout = 'tools/bin/node\n'
    await expect(ctx.subprocess.resolveExecutable('node')).resolves.toBe('/workspace/tools/bin/node')
    expect(fake.runs.at(-1)?.options.env).toBeUndefined()
  })

  it('rejects invalid executable lookup inputs and results', async () => {
    const fake = new FakeSandbox()
    const { ctx } = await service(fake)
    await expect(ctx.subprocess.resolveExecutable('')).rejects.toThrow('must be non-empty')
    await expect(ctx.subprocess.resolveExecutable('node', undefined, AbortSignal.abort(new Error('stop'))))
      .rejects.toThrow('stop')
    await expect(ctx.subprocess.resolveExecutable('./bin/server')).rejects.toThrow('is a relative path')
    fake.executableProbeExit = 1
    await expect(ctx.subprocess.resolveExecutable('/bin/missing')).rejects.toThrow('is not an executable file')
    fake.lookupExit = 1
    await expect(ctx.subprocess.resolveExecutable('node')).rejects.toThrow('was not found on PATH')
    fake.lookupExit = 0
    fake.lookupError = new Error('lookup transport failed')
    await expect(ctx.subprocess.resolveExecutable('node')).rejects.toThrow('lookup transport failed')
    fake.lookupError = undefined
    fake.lookupStdout = 'node\n'
    await expect(ctx.subprocess.resolveExecutable('node')).rejects.toThrow('did not resolve to one absolute path')
    fake.lookupStdout = '/one\n/two\n'
    await expect(ctx.subprocess.resolveExecutable('node')).rejects.toThrow('did not resolve to one absolute path')
  })

  it('terminates and joins live handles at disposal', async () => {
    const fake = new FakeSandbox()
    fake.trapsTerm = true
    const { ctx, fiber } = await service(fake)
    const handle = ctx.subprocess.spawn(spec({ graceMs: 1 }))
    await started(fake)
    await fiber.dispose()
    await expect(handle.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })
    expect(fake.groupLive).toBe(false)
  })

  it('rejects new spawns while disposing', async () => {
    const fake = new FakeSandbox()
    fake.trapsTerm = true
    const { ctx, fiber } = await service(fake)
    const subprocess = ctx.subprocess
    const handle = subprocess.spawn(spec({ graceMs: 1 }))
    await started(fake)
    const disposing = fiber.dispose()
    await flush()
    expect(() => subprocess.spawn(spec())).toThrow('service is disposing')
    await expect(subprocess.spawnTerminal({
      argv: ['bash'], cwd: '/workspace', rows: 24, cols: 80, graceMs: 5,
    })).rejects.toThrow('service is disposing')
    await disposing
    await handle.done
  })

  it('releases naturally settled handles before later disposal', async () => {
    const fake = new FakeSandbox()
    const { ctx, fiber } = await service(fake)
    const handle = ctx.subprocess.spawn(spec())
    await started(fake)
    await fake.completeBoth()
    fake.status = '0\n'
    fake.groupLive = false
    await handle.done
    await vi.waitFor(() => {
      expect((ctx.subprocess as unknown as { live: Set<unknown> }).live.size).toBe(0)
    })
    await fiber.dispose()
    expect(fake.signals).toEqual([])
  })

  it('retains a handle whose automatic release fails until disposal retries it', async () => {
    const fake = new FakeSandbox()
    fake.probeError = new Error('transient liveness failure')
    const { ctx, fiber } = await service(fake)
    const live = (ctx.subprocess as unknown as { live: Set<unknown> }).live
    const handle = ctx.subprocess.spawn(spec())
    await started(fake)
    await fake.completeBoth()
    fake.status = '0\n'
    await expect(handle.done).resolves.toEqual({ exitCode: 0, signal: null })
    await vi.waitFor(() => { expect(fake.groupProbes).toBeGreaterThan(0) })
    expect(live.size).toBe(1)

    fake.probeError = undefined
    fake.groupLive = false
    await fiber.dispose()
    expect(live.size).toBe(0)
  })

  it('joins a rejected command outcome while disposing its retained handle', async () => {
    const fake = new FakeSandbox()
    fake.statusError = new Error('status transport failed')
    fake.trapsTerm = true
    fake.survivesKill = true
    fake.killProcessError = new Error('daemon kill failed')
    const { ctx, fiber } = await service(fake)
    const live = (ctx.subprocess as unknown as { live: Set<unknown> }).live
    const handle = ctx.subprocess.spawn(spec({ graceMs: 1 }))
    await expect(handle.done).rejects.toThrow('process-group rollback did not reach quiescence')
    await vi.waitFor(() => { expect(live.size).toBe(1) })

    fake.survivesKill = false
    fake.killProcessError = undefined
    await fiber.dispose()
    expect(live.size).toBe(0)
  })

  it('reports a single disposal failure and aggregates siblings', async () => {
    const single = await service()
    const singleErrors: unknown[] = []
    single.ctx.logger.error = ((error: unknown) => { singleErrors.push(error) }) as typeof single.ctx.logger.error
    const live = (single.ctx.subprocess as unknown as { live: Set<unknown> }).live
    live.add({
      terminate: vi.fn(),
      waitForExit: vi.fn(async () => { throw new Error('only cleanup failed') }),
      done: Promise.resolve({ exitCode: 0, signal: null }),
    })
    await single.fiber.dispose()
    expect(singleErrors[0]).toMatchObject({ message: 'only cleanup failed' })

    const many = await service()
    const manyErrors: unknown[] = []
    many.ctx.logger.error = ((error: unknown) => { manyErrors.push(error) }) as typeof many.ctx.logger.error
    const handles = (many.ctx.subprocess as unknown as { live: Set<unknown> }).live
    handles.add({
      terminate: vi.fn(),
      waitForExit: vi.fn(async () => { throw new Error('first cleanup failed') }),
      done: Promise.resolve({ exitCode: 0, signal: null }),
    })
    handles.add({
      terminate: vi.fn(),
      waitForExit: vi.fn(async () => { throw new Error('second cleanup failed') }),
      done: Promise.resolve({ exitCode: 0, signal: null }),
    })
    await many.fiber.dispose()
    const failure = manyErrors[0]
    expect(failure).toBeInstanceOf(AggregateError)
    if (!(failure instanceof AggregateError)) throw new Error('expected AggregateError')
    expect(failure.errors.map(error => (error as Error).message).sort()).toEqual([
      'first cleanup failed',
      'second cleanup failed',
    ])
  })
})
