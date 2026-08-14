import { Buffer } from 'node:buffer'
import { once } from 'node:events'
import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, vi } from 'vitest'
import {
  RemoteAPIError,
  SandboxNotFoundError,
  type CommandResult,
  type CreatePtyOptions,
  type Pty,
  type RunOptions,
  type Sandbox,
} from '../src/runtime.ts'
import type TensorlakeRuntime from '../src/runtime.ts'
import type { SubprocessTerminalSpawnSpec } from '@deepseek-ai/dsh-subprocess'
import TensorlakeSubprocessRuntime from '../src/subprocess/index.ts'
import { MAX_TIMER_DELAY_MS } from '../src/subprocess/provider.ts'
import { spawnTensorlakeTerminal } from '../src/subprocess/terminal.ts'

const AMBIENT = 'PATH=/ambient/bin\0KEEP=safe\0UNICODE=你好\0NPM_TOKEN=secret\0DSH_STALE=old\0'

class FakePty {
  readonly inputs: string[] = []
  kills = 0
  killError: unknown
  sendInputError: unknown
  settleOnKill = true
  onData: ((data: Uint8Array) => void) | undefined
  private readonly exit = Promise.withResolvers<number>()
  private settled = false

  constructor() {
    void this.exit.promise.catch(() => {})
  }

  async sendInput(input: string | Uint8Array): Promise<void> {
    if (this.sendInputError !== undefined) throw this.sendInputError
    this.inputs.push(typeof input === 'string' ? input : Buffer.from(input).toString('utf8'))
  }

  wait(): Promise<number> {
    return this.exit.promise
  }

  async kill(): Promise<void> {
    this.kills += 1
    if (this.killError !== undefined) throw this.killError
    if (this.settleOnKill) this.exitWith(-1)
  }

  /** Settle the websocket transport's exit report; negative means a killed session. */
  exitWith(code: number): void {
    if (this.settled) return
    this.settled = true
    this.exit.resolve(code)
  }

  crash(error: unknown): void {
    if (this.settled) return
    this.settled = true
    this.exit.reject(error)
  }

  asPty(): Pty {
    return this as unknown as Pty
  }
}

class FakeTerminalSandbox {
  readonly pty = new FakePty()
  readonly runs: Array<{ command: string; args: string[] }> = []
  readonly commands: string[] = []
  readonly removed: string[] = []
  readonly prepared: string[] = []
  createOptions: CreatePtyOptions | undefined
  ambient = AMBIENT
  ambientError: unknown
  prepareExit = 0
  prepareError: unknown
  createError: unknown
  publishedPid = '4242'
  readonly pidQueue: string[] = []
  readFileError: unknown
  foreground = '4343\n'
  foregroundExit = 0
  foregroundError: unknown
  groups: number[] = [4242]
  sessionError: unknown
  signalError: unknown
  termError: unknown
  clearOnTerm = true
  clearOnKill = true
  removeExit = 0
  removeError: unknown
  afterPrepare: (() => void) | undefined
  afterKillSignal: (() => void) | undefined
  settlePtyOnSignal = true
  pidGate: Promise<undefined> | undefined
  prepareGate: Promise<undefined> | undefined
  private openPid: (() => void) | undefined
  private openPrepare: (() => void) | undefined

  deferPid(): void {
    const gate = Promise.withResolvers<undefined>()
    this.pidGate = gate.promise
    this.openPid = () => { gate.resolve(undefined) }
  }

  releasePid(): void {
    this.openPid?.()
  }

  deferPrepare(): void {
    const gate = Promise.withResolvers<undefined>()
    this.prepareGate = gate.promise
    this.openPrepare = () => { gate.resolve(undefined) }
  }

  releasePrepare(): void {
    this.openPrepare?.()
  }

  /** Deliver terminal bytes through the captured PTY data callback. */
  emit(text: string): void {
    this.createOptions?.onData?.(Buffer.from(text, 'utf8'))
  }

  readonly sandbox = {
    run: async (command: string, options: RunOptions = {}): Promise<CommandResult> => {
      const args = options.args ?? []
      this.runs.push({ command, args })
      if (command === 'rm') {
        this.removed.push(args.at(-1) ?? '')
        if (this.removeError !== undefined) throw this.removeError
        return { exitCode: this.removeExit, stdout: '', stderr: 'rm failed' }
      }
      const script = args[1] ?? ''
      this.commands.push(script.startsWith('kill -')
        ? `${(script.split(' ')[1] ?? '').slice(1)} ${args.slice(3).join(' ')}`
        : script)
      if (script.includes('env -0 | base64')) {
        if (this.ambientError !== undefined) throw this.ambientError
        return { exitCode: 0, stdout: Buffer.from(this.ambient, 'utf8').toString('base64'), stderr: '' }
      }
      if (script.includes('mkdir -p')) {
        this.prepared.push(args[3] ?? '')
        await this.prepareGate
        if (this.prepareError !== undefined) throw this.prepareError
        return { exitCode: this.prepareExit, stdout: '', stderr: 'prepare failed' }
      }
      if (script.includes('ps -eo sid=')) {
        if (this.sessionError !== undefined) throw this.sessionError
        return { exitCode: 0, stdout: this.groups.map(group => `${group}\n`).join(''), stderr: '' }
      }
      if (script.includes('ps -o tpgid=')) {
        if (this.foregroundError !== undefined) throw this.foregroundError
        return { exitCode: this.foregroundExit, stdout: this.foreground, stderr: '' }
      }
      if (script.startsWith('kill -')) {
        const signal = (script.split(' ')[1] ?? '').slice(1)
        if (signal === 'TERM' && this.termError !== undefined) throw this.termError
        if (this.signalError !== undefined) throw this.signalError
        if (signal === 'TERM' && this.clearOnTerm) this.settle()
        if (signal === 'KILL' && this.clearOnKill) this.settle()
        if (signal === 'KILL') this.afterKillSignal?.()
        return { exitCode: 0, stdout: '', stderr: '' }
      }
      return { exitCode: 0, stdout: '', stderr: '' }
    },
    readFile: async (_path: string): Promise<Uint8Array> => {
      await this.pidGate
      if (this.readFileError !== undefined) throw this.readFileError
      const raw = this.pidQueue.length > 0 ? this.pidQueue.shift() ?? '' : this.publishedPid
      return new TextEncoder().encode(raw)
    },
    createPty: async (options: CreatePtyOptions): Promise<Pty> => {
      this.createOptions = options
      this.pty.onData = options.onData
      if (this.createError !== undefined) throw this.createError
      return this.pty.asPty()
    },
  } as unknown as Sandbox

  private settle(): void {
    this.groups = []
    if (this.settlePtyOnSignal) this.pty.exitWith(-1)
  }
}

function runtime(fake: FakeTerminalSandbox): TensorlakeRuntime {
  return {
    cwd: '/workspace',
    runtimeRoot: '/workspace/.dsh-tensorlake',
    getSandbox: async () => fake.sandbox,
  } as unknown as TensorlakeRuntime
}

function spec(overrides: Partial<SubprocessTerminalSpawnSpec> = {}): SubprocessTerminalSpawnSpec {
  return {
    argv: ['/bin/bash', '--noprofile', '--norc'],
    cwd: '/workspace',
    rows: 24,
    cols: 80,
    graceMs: 5,
    env: { TERM: 'dumb', DSH_SESSION_ID: 'owner', TOKEN_EXPLICIT: 'kept' },
    ...overrides,
  }
}

/** Spawn the terminal under test with a short poll cadence. */
function testSpawn(
  provided: TensorlakeRuntime,
  request: SubprocessTerminalSpawnSpec,
  stateDir: string,
  pollMs = 1,
): ReturnType<typeof spawnTensorlakeTerminal> {
  return spawnTensorlakeTerminal(provided, request, stateDir, pollMs)
}

async function flush(): Promise<void> {
  await new Promise(resolve => setTimeout(resolve, 0))
}

describe('Tensorlake terminal allocation', () => {
  it('publishes the session leader pid and runs the requested argv under a rebuilt environment', async () => {
    const fake = new FakeTerminalSandbox()
    fake.pidQueue.push('', '4242')
    const terminal = await testSpawn(runtime(fake), spec(), '/runtime/terminal-one')
    expect(terminal.pid).toBe(4242)
    expect(fake.prepared).toEqual(['/runtime/terminal-one'])

    const args = fake.createOptions?.args ?? []
    const count = Number(args[4])
    expect(fake.createOptions).toMatchObject({ command: 'bash', workingDir: '/workspace', rows: 24, cols: 80 })
    expect(args[1]).toContain('exec env -i -- "${dsh_env[@]}" "$@"')
    expect(args[2]).toBe('dsh-subprocess-tensorlake')
    expect(args[3]).toBe('/runtime/terminal-one/pid')
    expect(args.slice(5, 5 + count)).toEqual([
      'PATH=/ambient/bin',
      'KEEP=safe',
      'UNICODE=你好',
      'TERM=dumb',
      'DSH_SESSION_ID=owner',
      'TOKEN_EXPLICIT=kept',
    ])
    expect(args.slice(5 + count)).toEqual(['/bin/bash', '--noprofile', '--norc'])

    let output = ''
    terminal.output.on('data', (chunk: Buffer) => { output += chunk.toString('utf8') })
    fake.emit('shell$ ')
    await flush()
    expect(output).toBe('shell$ ')

    await terminal.write('echo ok\r')
    expect(fake.pty.inputs).toEqual(['echo ok\r'])
    await expect(terminal.inspectForeground()).resolves.toEqual({ processGroupId: 4343, inputWaiting: false })
    await expect(terminal.signalForeground('SIGINT')).resolves.toBe(4343)
    expect(fake.commands).toContain('INT 4343')

    const terminating = terminal.terminate()
    await expect(terminal.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    await terminating
    expect(fake.commands).toContain('TERM -4242')
    expect(fake.removed).toEqual(['/runtime/terminal-one'])
    expect(fake.pty.kills).toBe(1)

    terminal.output.destroy()
    fake.emit('late bytes')
    expect(output).toBe('shell$ ')
  })

  it('rejects a terminal that exits or publishes an unusable process id', async () => {
    const exited = new FakeTerminalSandbox()
    exited.publishedPid = ''
    const exiting = testSpawn(runtime(exited), spec(), '/runtime/no-pid')
    await vi.waitFor(() => { expect(exited.createOptions).toBeDefined() })
    exited.pty.exitWith(0)
    await expect(exiting).rejects.toThrow('terminal exited before publishing its process id')
    expect(exited.pty.kills).toBe(1)
    expect(exited.removed).toEqual(['/runtime/no-pid'])

    const invalid = new FakeTerminalSandbox()
    invalid.publishedPid = 'not-a-pid'
    await expect(testSpawn(runtime(invalid), spec(), '/runtime/invalid-pid'))
      .rejects.toThrow('terminal published an invalid process id')

    const unsafe = new FakeTerminalSandbox()
    unsafe.publishedPid = '1'
    await expect(testSpawn(runtime(unsafe), spec(), '/runtime/unsafe-pid'))
      .rejects.toThrow('unsafe published terminal process id 1')
  })

  it('rolls back a canceled or failed allocation', async () => {
    const preAborted = new FakeTerminalSandbox()
    await expect(testSpawn(runtime(preAborted), spec({ signal: AbortSignal.abort(new Error('stop')) }), '/runtime/pre-abort'))
      .rejects.toThrow('stop')
    expect(preAborted.prepared).toEqual([])

    const ambient = new FakeTerminalSandbox()
    ambient.ambientError = new Error('ambient lookup failed')
    await expect(testSpawn(runtime(ambient), spec(), '/runtime/ambient'))
      .rejects.toThrow('ambient lookup failed')
    expect(ambient.removed).toEqual([])

    const invalidEnv = new FakeTerminalSandbox()
    await expect(testSpawn(runtime(invalidEnv), spec({ env: { 'BAD=NAME': 'x' } }), '/runtime/invalid-env'))
      .rejects.toThrow('environment entries require')
    expect(invalidEnv.prepared).toEqual([])

    const prepare = new FakeTerminalSandbox()
    prepare.prepareExit = 1
    await expect(testSpawn(runtime(prepare), spec(), '/runtime/prepare-failure'))
      .rejects.toThrow('bash exited 1')
    expect(prepare.removed).toEqual(['/runtime/prepare-failure'])
    expect(prepare.createOptions).toBeUndefined()

    const created = new FakeTerminalSandbox()
    created.createError = new Error('pty allocation failed')
    await expect(testSpawn(runtime(created), spec(), '/runtime/create-failure'))
      .rejects.toThrow('pty allocation failed')
    expect(created.removed).toEqual(['/runtime/create-failure'])
    expect(created.pty.kills).toBe(0)

    const expired = new FakeTerminalSandbox()
    expired.publishedPid = '1'
    expired.pty.killError = new SandboxNotFoundError('sandbox-gone')
    expired.removeError = new RemoteAPIError(404, 'missing')
    await expect(testSpawn(runtime(expired), spec(), '/runtime/expired-rollback'))
      .rejects.toThrow('unsafe published terminal process id 1')

    const failedCleanup = new FakeTerminalSandbox()
    failedCleanup.publishedPid = '1'
    failedCleanup.pty.killError = new Error('pty kill transport failed')
    failedCleanup.removeError = new Error('state removal transport failed')
    const failure = await testSpawn(runtime(failedCleanup), spec(), '/runtime/failed-rollback')
      .catch((error: unknown) => error as AggregateError)
    expect(failure).toBeInstanceOf(AggregateError)
    expect((failure as AggregateError).message).toContain('terminal setup cleanup did not complete')
    expect((failure as AggregateError).errors.map(error => (error as Error).message)).toEqual([
      'subprocess-tensorlake: unsafe published terminal process id 1',
      'pty kill transport failed',
      'state removal transport failed',
    ])
  })

  it('cancels a stalled allocation between remote round-trips', async () => {
    const fake = new FakeTerminalSandbox()
    fake.deferPrepare()
    const controller = new AbortController()
    const spawning = testSpawn(runtime(fake), spec({ signal: controller.signal }), '/runtime/stalled-allocation')
    await vi.waitFor(() => { expect(fake.prepared).toHaveLength(1) })
    controller.abort(new Error('allocation canceled'))
    fake.releasePrepare()
    await expect(spawning).rejects.toThrow('allocation canceled')
    expect(fake.createOptions).toBeUndefined()
    expect(fake.removed).toEqual(['/runtime/stalled-allocation'])

    const publishing = new FakeTerminalSandbox()
    publishing.pidQueue.push('')
    publishing.deferPid()
    const publishController = new AbortController()
    const publishSpawn = testSpawn(
      runtime(publishing),
      spec({ signal: publishController.signal }),
      '/runtime/stalled-publication',
    )
    await vi.waitFor(() => { expect(publishing.createOptions).toBeDefined() })
    publishController.abort(new Error('publication canceled'))
    publishing.releasePid()
    await expect(publishSpawn).rejects.toThrow('publication canceled')
    expect(publishing.pty.kills).toBe(1)
  })
})

describe('Tensorlake terminal lifecycle', () => {
  it('rejects operations after the terminal exits and after termination starts', async () => {
    const fake = new FakeTerminalSandbox()
    fake.groups = []
    const terminal = await testSpawn(runtime(fake), spec(), '/runtime/exited')
    terminal.output.resume()
    const ended = once(terminal.output, 'end')
    fake.pty.exitWith(7)
    await expect(terminal.done).resolves.toEqual({ exitCode: 7, signal: null })
    await ended

    await expect(terminal.write('late')).rejects.toThrow('terminal process has exited')
    fake.foregroundExit = 1
    fake.foreground = ''
    await expect(terminal.inspectForeground()).resolves.toBeUndefined()
    await expect(terminal.signalForeground('SIGINT')).rejects.toThrow('cannot resolve foreground process group')

    await terminal.terminate()
    const commands = fake.commands.length
    await expect(terminal.write('after termination')).rejects.toThrow('terminal is terminating')
    await expect(terminal.inspectForeground()).rejects.toThrow('terminal is terminating')
    await expect(terminal.signalForeground('SIGINT')).rejects.toThrow('terminal is terminating')
    expect(fake.commands).toHaveLength(commands)
  })

  it('joins in-flight operations before cleanup', async () => {
    const fake = new FakeTerminalSandbox()
    const terminal = await testSpawn(runtime(fake), spec(), '/runtime/in-flight')
    const release = Promise.withResolvers<undefined>()
    fake.pty.sendInput = async (input: string | Uint8Array): Promise<void> => {
      fake.pty.inputs.push(String(input))
      await release.promise
    }
    const write = terminal.write('slow input')
    await flush()
    const terminating = terminal.terminate()
    release.resolve(undefined)
    await write
    await terminating
    expect(fake.pty.inputs).toEqual(['slow input'])
  })

  it('reports foreground resolution failures raised while the terminal is live', async () => {
    const fake = new FakeTerminalSandbox()
    const terminal = await testSpawn(runtime(fake), spec(), '/runtime/foreground')
    fake.foreground = 'invalid\n'
    await expect(terminal.inspectForeground()).rejects.toThrow('cannot resolve foreground process group')
    fake.foregroundExit = 1
    await expect(terminal.inspectForeground()).rejects.toThrow('cannot resolve foreground process group')
    fake.foregroundExit = 0
    fake.foreground = '4242\n'
    await expect(terminal.signalForeground('SIGKILL')).rejects.toThrow('refusing to SIGKILL the terminal shell')
    await expect(terminal.signalForeground('SIGTERM')).resolves.toBe(4242)
    fake.foregroundError = new Error('foreground transport failed')
    await expect(terminal.inspectForeground()).rejects.toThrow('foreground transport failed')
    fake.foregroundError = undefined
    await terminal.terminate()
  })

  it('escalates a stubborn session to KILL and reports what survives', async () => {
    const survivor = new FakeTerminalSandbox()
    survivor.clearOnTerm = false
    survivor.clearOnKill = false
    const surviving = await testSpawn(runtime(survivor), spec({ graceMs: 0 }), '/runtime/surviving-groups')
    await expect(surviving.terminate()).rejects.toThrow('surviving process groups: 4242')
    expect(survivor.commands).toContain('TERM -4242')
    expect(survivor.commands).toContain('KILL -4242')

    survivor.clearOnKill = true
    await expect(surviving.terminate()).resolves.toBeUndefined()
    await expect(surviving.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })

    const livePid = new FakeTerminalSandbox()
    livePid.groups = []
    livePid.pty.settleOnKill = false
    const live = await testSpawn(runtime(livePid), spec({ graceMs: 1 }), '/runtime/surviving-pid')
    await expect(live.terminate()).rejects.toThrow('surviving pid: 4242')
    expect(livePid.pty.kills).toBe(1)
    livePid.pty.exitWith(0)
    await live.done
    await expect(live.terminate()).resolves.toBeUndefined()
  })

  it('waits out a session that empties inside the grace window', async () => {
    const fake = new FakeTerminalSandbox()
    fake.clearOnTerm = false
    const terminal = await testSpawn(runtime(fake), spec({ graceMs: 50 }), '/runtime/slow-exit')
    setTimeout(() => {
      fake.groups = []
      fake.pty.exitWith(-1)
    }, 5)
    await expect(terminal.terminate()).resolves.toBeUndefined()
    await expect(terminal.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
  })

  it('tolerates a disappeared sandbox and propagates other cleanup failures', async () => {
    const expired = new FakeTerminalSandbox()
    expired.groups = []
    expired.pty.settleOnKill = false
    expired.pty.killError = new SandboxNotFoundError('sandbox-gone')
    const expiredTerminal = await testSpawn(runtime(expired), spec({ graceMs: 1 }), '/runtime/expired-kill')
    await expect(expiredTerminal.terminate()).rejects.toThrow('surviving pid: 4242')
    expect(expired.pty.kills).toBe(1)
    expired.pty.exitWith(0)
    await expiredTerminal.done

    const failed = new FakeTerminalSandbox()
    failed.groups = []
    failed.pty.settleOnKill = false
    failed.pty.killError = new Error('pty kill transport failed')
    const failedTerminal = await testSpawn(runtime(failed), spec({ graceMs: 1 }), '/runtime/failed-kill')
    await expect(failedTerminal.terminate()).rejects.toThrow('pty kill transport failed')
    failed.pty.killError = undefined
    failed.pty.exitWith(0)
    await failedTerminal.done
    await expect(failedTerminal.terminate()).resolves.toBeUndefined()

    const signalFailure = new FakeTerminalSandbox()
    signalFailure.termError = new Error('signal transport failed')
    const signalTerminal = await testSpawn(runtime(signalFailure), spec({ graceMs: 1 }), '/runtime/failed-signal')
    await expect(signalTerminal.terminate()).rejects.toThrow('signal transport failed')
    signalFailure.termError = undefined
    await expect(signalTerminal.terminate()).resolves.toBeUndefined()

    const residue = new FakeTerminalSandbox()
    residue.removeError = new Error('state removal failed')
    const residueTerminal = await testSpawn(runtime(residue), spec(), '/runtime/residue')
    await expect(residueTerminal.terminate()).resolves.toBeUndefined()
    expect(residue.removed).toEqual(['/runtime/residue'])
  })

  it('stops waiting for a terminal whose transport rejects during cleanup', async () => {
    const quiet = new FakeTerminalSandbox()
    quiet.groups = []
    const quietTerminal = await testSpawn(runtime(quiet), spec({ graceMs: 200 }), '/runtime/reject-while-waiting')
    const quietFailure = expect(quietTerminal.done).rejects.toThrow('terminal transport failed')
    quietTerminal.output.on('error', () => {})
    setTimeout(() => { quiet.pty.crash(new Error('terminal transport failed')) }, 5)
    await expect(quietTerminal.terminate()).resolves.toBeUndefined()
    await quietFailure

    const stubborn = new FakeTerminalSandbox()
    stubborn.clearOnTerm = false
    stubborn.settlePtyOnSignal = false
    stubborn.pty.settleOnKill = false
    // The session empties under force while the leader is still live, so
    // cleanup waits on the outcome that the transport then rejects.
    stubborn.afterKillSignal = () => {
      stubborn.afterKillSignal = undefined
      setTimeout(() => { stubborn.pty.crash(new Error('late transport failure')) }, 5)
    }
    const stubbornTerminal = await testSpawn(runtime(stubborn), spec({ graceMs: 50 }), '/runtime/reject-after-kill')
    const stubbornFailure = expect(stubbornTerminal.done).rejects.toThrow('late transport failure')
    stubbornTerminal.output.on('error', () => {})
    await expect(stubbornTerminal.terminate()).resolves.toBeUndefined()
    await stubbornFailure
    expect(stubborn.commands).toContain('KILL -4242')
  })

  it('deletes the settled PTY session and reports a failure that is not a missing sandbox', async () => {
    const missing = new FakeTerminalSandbox()
    missing.groups = []
    const tolerant = await testSpawn(runtime(missing), spec(), '/runtime/final-kill-missing')
    missing.pty.exitWith(0)
    await tolerant.done
    missing.pty.killError = new SandboxNotFoundError('sandbox-gone')
    await expect(tolerant.terminate()).resolves.toBeUndefined()
    expect(missing.pty.kills).toBe(1)

    const failed = new FakeTerminalSandbox()
    failed.groups = []
    const terminal = await testSpawn(runtime(failed), spec(), '/runtime/final-kill-failure')
    failed.pty.exitWith(0)
    await terminal.done
    failed.pty.killError = new Error('session delete failed')
    await expect(terminal.terminate()).rejects.toThrow('session delete failed')
  })

  it('maps the websocket exit report to the requested termination signal', async () => {
    const killed = new FakeTerminalSandbox()
    killed.groups = []
    const unrequested = await testSpawn(runtime(killed), spec(), '/runtime/unrequested-kill')
    killed.pty.exitWith(-1)
    await expect(unrequested.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })
    await unrequested.terminate()

    const requested = new FakeTerminalSandbox()
    requested.clearOnTerm = false
    requested.clearOnKill = false
    const terminal = await testSpawn(runtime(requested), spec({ graceMs: 0 }), '/runtime/requested-exit')
    const terminating = terminal.terminate().catch(() => undefined)
    await vi.waitFor(() => { expect(requested.commands).toContain('KILL -4242') })
    requested.groups = []
    requested.pty.exitWith(0)
    await expect(terminal.done).resolves.toEqual({ exitCode: null, signal: 'SIGKILL' })
    await terminating
  })

  it('destroys terminal output when the transport itself fails', async () => {
    const fake = new FakeTerminalSandbox()
    fake.groups = []
    const terminal = await testSpawn(runtime(fake), spec(), '/runtime/transport-failure')
    const failure = once(terminal.output, 'error')
    const settled = expect(terminal.done).rejects.toThrow('terminal transport failed')
    fake.pty.crash(new Error('terminal transport failed'))
    await settled
    await expect(failure).resolves.toMatchObject([{ message: 'terminal transport failed' }])
    await terminal.terminate()
  })
})

describe('Tensorlake subprocess terminal service', () => {
  async function service(fake = new FakeTerminalSandbox()): Promise<{
    ctx: Context
    fiber: Awaited<ReturnType<Context['plugin']>>
    fake: FakeTerminalSandbox
  }> {
    const ctx = new Context()
    ctx.provide('tensorlake', runtime(fake) as never)
    const fiber = await ctx.plugin(TensorlakeSubprocessRuntime, { pollMs: 1 })
    return { ctx, fiber, fake }
  }

  it('validates terminal requests before any remote work', async () => {
    const { ctx, fake } = await service()
    await expect(ctx.subprocess.spawnTerminal(spec({ argv: [] }))).rejects.toThrow('terminal argv must contain a program')
    await expect(ctx.subprocess.spawnTerminal(spec({ argv: [''] }))).rejects.toThrow('terminal argv must contain a program')
    for (const graceMs of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, MAX_TIMER_DELAY_MS + 1]) {
      await expect(ctx.subprocess.spawnTerminal(spec({ graceMs })))
        .rejects.toThrow('graceMs must be a positive finite number')
    }
    await expect(ctx.subprocess.spawnTerminal(spec({ signal: AbortSignal.abort(new Error('canceled')) })))
      .rejects.toThrow('canceled')
    expect(fake.createOptions).toBeUndefined()
  })

  it('owns live terminals through service disposal', async () => {
    const { ctx, fiber, fake } = await service()
    const terminal = await ctx.subprocess.spawnTerminal(spec({ signal: new AbortController().signal }))
    await fiber.dispose()
    await expect(terminal.done).resolves.toEqual({ exitCode: null, signal: 'SIGTERM' })
    expect(fake.removed).toEqual([expect.stringContaining('/terminals/')])
  })

  it('releases naturally settled terminals before later disposal', async () => {
    const { ctx, fiber, fake } = await service()
    fake.groups = []
    const terminal = await ctx.subprocess.spawnTerminal(spec())
    fake.pty.exitWith(0)
    await terminal.done
    await vi.waitFor(() => {
      expect((ctx.subprocess as unknown as { terminals: Set<unknown> }).terminals.size).toBe(0)
    })
    const commands = fake.commands.length
    await fiber.dispose()
    expect(fake.commands).toHaveLength(commands)
  })

  it('retains a terminal whose automatic release fails until disposal retries it', async () => {
    const { ctx, fiber, fake } = await service()
    fake.clearOnTerm = false
    fake.clearOnKill = false
    const terminals = (ctx.subprocess as unknown as { terminals: Set<unknown> }).terminals
    const terminal = await ctx.subprocess.spawnTerminal(spec({ graceMs: 1 }))
    fake.pty.exitWith(0)
    await terminal.done
    await vi.waitFor(() => { expect(fake.commands).toContain('KILL -4242') })
    expect(terminals.size).toBe(1)

    fake.groups = []
    await fiber.dispose()
    expect(terminals.size).toBe(0)
  })

  it('rejects and rolls back terminal setup that completes during disposal', async () => {
    const { ctx, fiber, fake } = await service()
    fake.deferPid()
    const spawning = ctx.subprocess.spawnTerminal(spec())
    const rejected = expect(spawning).rejects.toThrow('service disposed during terminal setup')
    await vi.waitFor(() => { expect(fake.createOptions).toBeDefined() })

    const disposing = fiber.dispose()
    await flush()
    fake.releasePid()
    await rejected
    await disposing
    expect(fake.groups).toEqual([])
    expect(fake.removed).toEqual([expect.stringContaining('/terminals/')])
  })

  it('aborts terminal setup that has not reached its remote allocation', async () => {
    const { ctx, fiber, fake } = await service()
    fake.deferPrepare()
    const spawning = ctx.subprocess.spawnTerminal(spec())
    const rejected = expect(spawning).rejects.toThrow('service disposed during terminal setup')
    await vi.waitFor(() => { expect(fake.prepared).toHaveLength(1) })

    const disposing = fiber.dispose()
    await flush()
    fake.releasePrepare()
    await rejected
    await disposing
    expect(fake.createOptions).toBeUndefined()
    expect(fake.removed).toEqual([expect.stringContaining('/terminals/')])
  })
})
