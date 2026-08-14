import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Mock } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import type { CommandResult, RunOptions, Sandbox as SandboxType } from 'tensorlake'
import TensorlakeRuntime, {
  isRemoteMissing,
  RemoteAPIError,
  runChecked,
  SandboxError,
  SandboxNotFoundError,
  TensorlakeCommandError,
} from '../src/runtime.ts'

const CWD = '/workspace'
const RUNTIME_ROOT = '/workspace/.dsh-tensorlake'

const sdk = vi.hoisted(() => ({
  create: vi.fn(),
}))

vi.mock('tensorlake', async (importOriginal) => {
  const actual = await importOriginal<typeof import('tensorlake')>()
  // The mock replaces only the SDK's static factory surface and is never constructed.
  class FakeSandbox {
    static create(...args: unknown[]): unknown {
      return sdk.create(...args)
    }
  }
  return { ...actual, Sandbox: FakeSandbox }
})

type RunCommand = (command: string, options?: RunOptions) => Promise<CommandResult>

interface SandboxFixture {
  sandbox: SandboxType
  run: Mock<RunCommand>
  terminate: ReturnType<typeof vi.fn>
}

function commandResult(overrides: Partial<CommandResult> = {}): CommandResult {
  return { exitCode: 0, stdout: '', stderr: '', ...overrides }
}

function fakeSandbox(id = 'sandbox-1'): SandboxFixture {
  const run = vi.fn<RunCommand>().mockResolvedValue(commandResult())
  const terminate = vi.fn().mockResolvedValue(undefined)
  const sandbox = { sandboxId: id, run, terminate } as unknown as SandboxType
  return { sandbox, run, terminate }
}

beforeEach(() => {
  sdk.create.mockReset()
  vi.unstubAllEnvs()
})

describe('TensorlakeRuntime', () => {
  it('creates one shared sandbox with a private runtime root and terminates it on disposal', async () => {
    const fixture = fakeSandbox()
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

    const service = ctx.tensorlake
    await expect(service.getSandbox()).resolves.toBe(fixture.sandbox)
    expect(service.cwd).toBe(CWD)
    expect(service.runtimeRoot).toBe(RUNTIME_ROOT)
    expect(sdk.create).toHaveBeenCalledWith({ apiKey: 'test-key', timeoutSecs: 600 })
    const createCwd = fixture.run.mock.calls[0]
    expect(createCwd?.[0]).toBe('bash')
    expect(createCwd?.[1]?.args?.[1]).toContain('sudo -n mkdir -p -- "$1"')
    expect(createCwd?.[1]?.args?.at(-1)).toBe(CWD)
    expect(fixture.run).toHaveBeenNthCalledWith(2, 'mkdir', { args: ['-p', '--', RUNTIME_ROOT] })
    expect(fixture.run).toHaveBeenNthCalledWith(3, 'bash', {
      args: ['-c', 'test -d "$1" && ! test -L "$1"', 'dsh-tensorlake', RUNTIME_ROOT],
    })
    expect(fixture.run).toHaveBeenNthCalledWith(4, 'chmod', { args: ['700', '--', RUNTIME_ROOT] })

    await fiber.dispose()
    expect(fixture.terminate).toHaveBeenCalledOnce()
    await expect(service.getSandbox()).rejects.toThrow(/disposing/)
  })

  it('reads the key from the environment and forwards the configured lifetime and allocations', async () => {
    vi.stubEnv('TENSORLAKE_API_KEY', 'environment-key')
    const fixture = fakeSandbox('configured-sandbox')
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, {
      cwd: '/workspace/project',
      timeoutSecs: 60,
      cpus: 2,
      memoryMb: 2048,
      diskMb: 8192,
    })
    await ctx.tensorlake.getSandbox()

    expect(sdk.create).toHaveBeenCalledWith({
      apiKey: 'environment-key',
      timeoutSecs: 60,
      cpus: 2,
      memoryMb: 2048,
      diskMb: 8192,
    })
    expect(ctx.tensorlake.cwd).toBe('/workspace/project')
    expect(ctx.tensorlake.runtimeRoot).toBe('/workspace/project/.dsh-tensorlake')
    expect(fixture.run.mock.calls[0]?.[0]).toBe('bash')
    expect(fixture.run.mock.calls[0]?.[1]?.args?.at(-1)).toBe('/workspace/project')

    await fiber.dispose()
    expect(fixture.terminate).toHaveBeenCalledOnce()
  })

  it('rejects handle acquisition when disposal starts during setup', async () => {
    const fixture = fakeSandbox()
    const opening = Promise.withResolvers<SandboxType>()
    sdk.create.mockReturnValue(opening.promise)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

    const acquisition = ctx.tensorlake.getSandbox()
    const disposing = fiber.dispose()
    opening.resolve(fixture.sandbox)

    await expect(acquisition).rejects.toThrow(/disposing/)
    await expect(disposing).resolves.toBeUndefined()
    expect(fixture.terminate).toHaveBeenCalledOnce()
  })

  it.each([
    ['the typed lifecycle miss', new SandboxNotFoundError('sandbox-1')],
    ['the proxy 404 report', new RemoteAPIError(404, 'sandbox gone')],
  ])('accepts %s when disposal terminates an already-removed sandbox', async (_label, missing) => {
    const fixture = fakeSandbox()
    fixture.terminate.mockRejectedValue(missing)
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const errors: unknown[] = []
    ctx.logger.error = ((error: unknown) => { errors.push(error) }) as typeof ctx.logger.error
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })
    await ctx.tensorlake.getSandbox()

    await expect(fiber.dispose()).resolves.toBeUndefined()
    expect(fixture.terminate).toHaveBeenCalledOnce()
    expect(errors).toEqual([])
  })

  it('does not classify other disposal failures as an already-gone sandbox', async () => {
    const fixture = fakeSandbox()
    const failure = new RemoteAPIError(500, 'disposition unknown')
    fixture.terminate.mockRejectedValue(failure)
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const errors: unknown[] = []
    ctx.logger.error = ((error: unknown) => { errors.push(error) }) as typeof ctx.logger.error
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })
    await ctx.tensorlake.getSandbox()

    await expect(fiber.dispose()).resolves.toBeUndefined()
    expect(fixture.terminate).toHaveBeenCalledOnce()
    expect(errors).toContain(failure)
  })

  it('retries the first control command while the sandbox proxy is still cold', async () => {
    vi.useFakeTimers()
    try {
      const fixture = fakeSandbox()
      fixture.run.mockRejectedValueOnce(new SandboxError('proxy route not ready'))
      sdk.create.mockResolvedValue(fixture.sandbox)
      const ctx = new Context()
      const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

      const acquisition = ctx.tensorlake.getSandbox()
      await vi.advanceTimersByTimeAsync(1_000)

      await expect(acquisition).resolves.toBe(fixture.sandbox)
      expect(fixture.run.mock.calls.map(call => call[0])).toEqual(['bash', 'bash', 'mkdir', 'bash', 'chmod'])
      expect(fixture.run.mock.calls[1]?.[1]?.args?.at(-1)).toBe(CWD)
      expect(fixture.terminate).not.toHaveBeenCalled()

      await fiber.dispose()
    } finally {
      vi.useRealTimers()
    }
  })

  it('gives up on a cold proxy once the readiness deadline passes', async () => {
    vi.useFakeTimers()
    try {
      const fixture = fakeSandbox()
      fixture.run.mockRejectedValue(new Error('fetch failed'))
      sdk.create.mockResolvedValue(fixture.sandbox)
      const ctx = new Context()
      const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

      // The rejection lands inside the timer advance, so the assertion must observe it first.
      const acquisition = expect(ctx.tensorlake.getSandbox()).rejects.toThrow('fetch failed')
      await vi.advanceTimersByTimeAsync(31_000)

      await acquisition
      expect(fixture.run.mock.calls.length).toBeGreaterThan(1)
      expect(fixture.terminate).toHaveBeenCalledOnce()

      await fiber.dispose()
      expect(fixture.terminate).toHaveBeenCalledOnce()
    } finally {
      vi.useRealTimers()
    }
  })

  it.each([
    ['a proxy-reported API failure', (fixture: SandboxFixture): void => {
      fixture.run.mockRejectedValueOnce(new RemoteAPIError(502, 'bad gateway'))
    }, 'bad gateway'],
    ['a nonzero exit', (fixture: SandboxFixture): void => {
      fixture.run.mockResolvedValueOnce(commandResult({ exitCode: 1, stderr: 'read-only file system' }))
    }, 'bash exited 1: read-only file system'],
  ])('never retries %s from the reachable sandbox daemon', async (_label, arrange, message) => {
    const fixture = fakeSandbox()
    arrange(fixture)
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

    await expect(ctx.tensorlake.getSandbox()).rejects.toThrow(message)
    expect(fixture.run).toHaveBeenCalledOnce()
    expect(fixture.terminate).toHaveBeenCalledOnce()
    await fiber.dispose()
  })

  it('rejects a reserved runtime root that is not a real directory', async () => {
    const fixture = fakeSandbox()
    fixture.run.mockImplementation((command, options) =>
      // Only the directory probe fails; the cwd-creation bash script succeeds.
      Promise.resolve(commandResult(command === 'bash' && options?.args?.[1]?.includes('test -d') === true
        ? { exitCode: 1 }
        : {})))
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

    await expect(ctx.tensorlake.getSandbox()).rejects.toThrow('runtime root must be a real directory')
    expect(fixture.run.mock.calls.map(call => call[0])).toEqual(['bash', 'mkdir', 'bash'])
    expect(fixture.terminate).toHaveBeenCalledOnce()
    await fiber.dispose()
  })

  it('preserves the setup failure after its one rollback attempt fails', async () => {
    const fixture = fakeSandbox()
    fixture.run.mockImplementation(command =>
      Promise.resolve(commandResult(command === 'chmod' ? { exitCode: 1, stderr: 'operation not permitted' } : {})))
    fixture.terminate.mockRejectedValueOnce(new Error('cleanup failed'))
    sdk.create.mockResolvedValue(fixture.sandbox)
    const ctx = new Context()
    const fiber = await ctx.plugin(TensorlakeRuntime, { apiKey: 'test-key' })

    await expect(ctx.tensorlake.getSandbox()).rejects.toThrow('chmod exited 1: operation not permitted')
    expect(fixture.terminate).toHaveBeenCalledOnce()

    await fiber.dispose()
    expect(fixture.terminate).toHaveBeenCalledOnce()
  })

  it.each([
    [{ apiKey: '' }, /configure apiKey/],
    [{ apiKey: 'x', cwd: 'relative' }, /absolute Linux path/],
    [{ apiKey: 'x', timeoutSecs: 0 }, /positive integer/],
    [{ apiKey: 'x', timeoutSecs: 1.5 }, /positive integer/],
    [{ apiKey: 'x', cpus: 0 }, /cpus must be a positive finite number/],
    [{ apiKey: 'x', memoryMb: Number.POSITIVE_INFINITY }, /memoryMb must be a positive finite number/],
    [{ apiKey: 'x', diskMb: -1 }, /diskMb must be a positive finite number/],
  ] as const)('fails self-contained configuration before opening Tensorlake: %j', async (config, message) => {
    vi.stubEnv('TENSORLAKE_API_KEY', '')
    const ctx = new Context()
    await expect(ctx.plugin(TensorlakeRuntime, config)).rejects.toThrow(message)
    expect(sdk.create).not.toHaveBeenCalled()
  })

  it('requires a key when both config and the environment omit it', async () => {
    const original = process.env.TENSORLAKE_API_KEY
    delete process.env.TENSORLAKE_API_KEY
    try {
      const ctx = new Context()
      await expect(ctx.plugin(TensorlakeRuntime, {})).rejects.toThrow(/configure apiKey/)
    } finally {
      if (original === undefined) delete process.env.TENSORLAKE_API_KEY
      else process.env.TENSORLAKE_API_KEY = original
    }
  })
})

describe('Tensorlake helpers', () => {
  it.each([
    ['the typed lifecycle miss', new SandboxNotFoundError('sandbox-1'), true],
    ['the proxy 404 report', new RemoteAPIError(404, 'gone'), true],
    ['another API status', new RemoteAPIError(500, 'server error'), false],
    ['an unrelated rejection', new Error('socket hang up'), false],
  ])('reports %s as a remote miss: %s', (_label, error, expected) => {
    expect(isRemoteMissing(error)).toBe(expected)
  })

  it('returns the result of a zero-exit command with its working directory and environment', async () => {
    const fixture = fakeSandbox()
    const result = commandResult({ stdout: 'ok\n' })
    fixture.run.mockResolvedValue(result)

    await expect(runChecked(fixture.sandbox, 'ls', ['-la'], { workingDir: '/tmp', env: { LANG: 'C' } }))
      .resolves.toBe(result)
    expect(fixture.run).toHaveBeenCalledWith('ls', {
      args: ['-la'],
      workingDir: '/tmp',
      env: { LANG: 'C' },
    })
  })

  it.each([
    ['the trimmed stderr', { exitCode: 2, stdout: 'ignored\n', stderr: ' denied\n' }, 'dsh-tensorlake: rm exited 2: denied'],
    ['the stdout when stderr is empty', { exitCode: 3, stdout: 'partial\n', stderr: '' }, 'dsh-tensorlake: rm exited 3: partial'],
    ['a placeholder when both streams are empty', { exitCode: 4, stdout: '', stderr: '' }, 'dsh-tensorlake: rm exited 4: no output'],
  ])('reports a nonzero exit with %s', async (_label, result, message) => {
    const fixture = fakeSandbox()
    fixture.run.mockResolvedValue(result)

    const error = await runChecked(fixture.sandbox, 'rm', ['-rf', '--', '/tmp/x']).catch((cause: unknown) => cause)
    expect(error).toBeInstanceOf(TensorlakeCommandError)
    expect((error as TensorlakeCommandError).name).toBe('TensorlakeCommandError')
    expect((error as TensorlakeCommandError).message).toBe(message)
    expect((error as TensorlakeCommandError).result).toBe(result)
    expect(fixture.run).toHaveBeenCalledWith('rm', { args: ['-rf', '--', '/tmp/x'] })
  })
})
