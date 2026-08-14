/**
 * Shared ownership of one Tensorlake sandbox. Capability adapters await the
 * same SDK handle, so filesystem and process operations inhabit one remote
 * Linux world.
 * @module @tensorlake/dsh-sandbox/runtime
 */

import { posix } from 'node:path'
import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { RemoteAPIError, Sandbox, SandboxNotFoundError } from 'tensorlake'
import type { CommandResult, RunOptions } from 'tensorlake'

/** How long the owner keeps absorbing proxy cold-start transport failures after creation. */
const PROXY_READY_DEADLINE_MS = 30_000
const PROXY_READY_RETRY_MS = 1_000

/**
 * Create the shared working directory. The sandbox daemon runs commands as
 * the unprivileged image user, so a cwd outside user-writable paths (the
 * loader smoke mirrors a host temp directory) escalates through the image's
 * passwordless sudo and hands the leaf back to the user.
 */
const CREATE_CWD_SCRIPT = [
  'if mkdir -p -- "$1" 2>/dev/null && test -w "$1"; then exit 0; fi',
  'sudo -n mkdir -p -- "$1" && sudo -n chown -- "$(id -u):$(id -g)" "$1" && test -w "$1"',
].join('\n')

export {
  OutputMode,
  ProcessStatus,
  Pty,
  RemoteAPIError,
  Sandbox,
  SandboxConnectionError,
  SandboxError,
  SandboxNotFoundError,
  StdinMode,
} from 'tensorlake'
export type {
  CommandResult,
  CreatePtyOptions,
  DirectoryEntry,
  ListDirectoryResponse,
  OutputEvent,
  OutputResponse,
  ProcessInfo,
  RunOptions,
  StartProcessOptions,
} from 'tensorlake'

/**
 * Report whether an SDK failure means the addressed sandbox, path, or process
 * no longer exists: the lifecycle API's typed miss or the sandbox proxy's
 * HTTP 404 report.
 * @param error - Any SDK rejection.
 * @returns `true` for a typed or proxy-reported miss, `false` otherwise.
 */
export function isRemoteMissing(error: unknown): boolean {
  return error instanceof SandboxNotFoundError
    || (error instanceof RemoteAPIError && error.statusCode === 404)
}

/** Nonzero exit of one adapter-internal control command, with its captured streams. */
export class TensorlakeCommandError extends Error {
  /**
   * Wrap one failed control command.
   * @param command - The argv[0] that ran.
   * @param result - The complete captured command result.
   */
  constructor(command: string, readonly result: CommandResult) {
    super(`dsh-tensorlake: ${command} exited ${result.exitCode}: ${result.stderr.trim() || result.stdout.trim() || 'no output'}`)
    this.name = 'TensorlakeCommandError'
  }
}

/**
 * Run one argv-style control command (never shell-interpreted) and require a
 * zero exit. Tensorlake's `run` resolves with the exit code instead of
 * throwing, so adapter-internal commands share this single check.
 * @param sandbox - Live SDK handle.
 * @param command - Executable name or absolute path.
 * @param args - Exact argument vector; the daemon passes it without a shell.
 * @param options - Optional working directory and environment overlay.
 * @returns the complete result of the zero-exit command.
 * @throws {TensorlakeCommandError} for a nonzero exit.
 */
export async function runChecked(
  sandbox: Sandbox,
  command: string,
  args: readonly string[],
  options: Pick<RunOptions, 'env' | 'workingDir'> = {},
): Promise<CommandResult> {
  const result = await sandbox.run(command, { args: [...args], ...options })
  if (result.exitCode !== 0) throw new TensorlakeCommandError(command, result)
  return result
}

/** Configuration for the shared Tensorlake sandbox owner. */
export interface Config {
  /** API key; omission reads `TENSORLAKE_API_KEY`. It is never forwarded into the sandbox. */
  apiKey?: string
  /** Shared remote working directory, created before adapters receive the sandbox. */
  cwd?: string
  /** Sandbox inactivity timeout in seconds; expiry terminates the unnamed sandbox. */
  timeoutSecs?: number
  /** Virtual CPU allocation; omission uses the service default. */
  cpus?: number
  /** Memory allocation in megabytes; omission uses the service default. */
  memoryMb?: number
  /** Root disk allocation in megabytes; omission uses the service default. */
  diskMb?: number
}

interface ResolvedConfig {
  apiKey: string
  cwd: string
  timeoutSecs: number
  cpus?: number
  memoryMb?: number
  diskMb?: number
}

interface SchemaResolvedConfig extends Config {
  cwd: string
  timeoutSecs: number
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    tensorlake: TensorlakeRuntime
  }
}

/**
 * Creates one lazily consumable Tensorlake SDK handle and terminates the
 * sandbox at timeout or disposal. Creation begins at plugin construction;
 * adapters await {@link getSandbox} before their first operation.
 */
export class TensorlakeRuntime extends Service {
  static Config: z<Config> = z.object({
    apiKey: z.string(),
    cwd: z.string().default('/home/tl-user/workspace'),
    timeoutSecs: z.number().default(600),
    cpus: z.number(),
    memoryMb: z.number(),
    diskMb: z.number(),
  })

  /** Validated remote working directory shared by provider adapters. */
  readonly cwd: string
  /** Remote directory reserved for adapter-owned process and terminal state. */
  readonly runtimeRoot: string

  private readonly config: ResolvedConfig
  private readonly ready: Promise<Sandbox>
  private disposed = false

  constructor(ctx: Context, config: Config) {
    super(ctx, 'tensorlake')
    // Schemastery fills the defaulted fields before construction; the type does not encode that step.
    const resolved = config as SchemaResolvedConfig
    const apiKey = config.apiKey ?? process.env.TENSORLAKE_API_KEY
    this.config = {
      apiKey: apiKey ?? '',
      cwd: resolved.cwd,
      timeoutSecs: resolved.timeoutSecs,
      ...(resolved.cpus !== undefined ? { cpus: resolved.cpus } : {}),
      ...(resolved.memoryMb !== undefined ? { memoryMb: resolved.memoryMb } : {}),
      ...(resolved.diskMb !== undefined ? { diskMb: resolved.diskMb } : {}),
    }
    this.validate()
    this.cwd = this.config.cwd
    this.runtimeRoot = posix.join(this.cwd, '.dsh-tensorlake')
    this.ready = this.open()
    // A deployment may load the owner before any adapter uses it. Keep a
    // failed eager connection observed; getSandbox() still returns the error.
    void this.ready.catch(() => {})
    ctx.effect(() => this.teardown, 'tensorlake sandbox teardown')
  }

  /**
   * Return the shared live SDK handle.
   * @returns the created sandbox after the configured cwd exists.
   * @throws when Tensorlake rejects creation or the service is disposing.
   */
  async getSandbox(): Promise<Sandbox> {
    this.assertNotDisposing()
    const sandbox = await this.ready
    // Disposal can race the awaited sandbox readiness despite the synchronous precheck.
    this.assertNotDisposing()
    return sandbox
  }

  private assertNotDisposing(): void {
    if (this.disposed) throw new Error('Tensorlake sandbox service is disposing')
  }

  private readonly teardown = async (): Promise<void> => {
    this.disposed = true
    let sandbox: Sandbox
    try {
      sandbox = await this.ready
    } catch {
      // open() either acquired no sandbox or already made its one rollback attempt.
      return
    }
    try {
      await sandbox.terminate()
    } catch (error: unknown) {
      if (!isRemoteMissing(error)) throw error
    }
  }

  private validate(): void {
    if (this.config.apiKey.length === 0) {
      throw new Error('dsh-tensorlake: configure apiKey or set TENSORLAKE_API_KEY')
    }
    if (!posix.isAbsolute(this.config.cwd)) {
      throw new Error(`dsh-tensorlake: cwd must be an absolute Linux path: ${this.config.cwd}`)
    }
    if (!Number.isSafeInteger(this.config.timeoutSecs) || this.config.timeoutSecs <= 0) {
      throw new Error('dsh-tensorlake: timeoutSecs must be a positive integer')
    }
    for (const [name, value] of Object.entries({
      cpus: this.config.cpus,
      memoryMb: this.config.memoryMb,
      diskMb: this.config.diskMb,
    })) {
      if (value !== undefined && (!Number.isFinite(value) || value <= 0)) {
        throw new Error(`dsh-tensorlake: ${name} must be a positive finite number`)
      }
    }
  }

  private async open(): Promise<Sandbox> {
    const sandbox = await Sandbox.create({
      apiKey: this.config.apiKey,
      timeoutSecs: this.config.timeoutSecs,
      ...(this.config.cpus !== undefined ? { cpus: this.config.cpus } : {}),
      ...(this.config.memoryMb !== undefined ? { memoryMb: this.config.memoryMb } : {}),
      ...(this.config.diskMb !== undefined ? { diskMb: this.config.diskMb } : {}),
    })
    try {
      // Creation returns when the sandbox is Running, but the proxy route can
      // lag briefly; absorb transport failures on the first control command
      // only. A daemon-reported failure (RemoteAPIError or a nonzero exit)
      // proves the proxy is reachable and is never retried.
      const deadline = Date.now() + PROXY_READY_DEADLINE_MS
      for (;;) {
        try {
          await runChecked(sandbox, 'bash', ['-c', CREATE_CWD_SCRIPT, 'dsh-tensorlake', this.cwd])
          break
        } catch (error: unknown) {
          if (error instanceof RemoteAPIError || error instanceof TensorlakeCommandError
            || Date.now() >= deadline) {
            throw error
          }
          await new Promise(resolve => setTimeout(resolve, PROXY_READY_RETRY_MS))
        }
      }
      await runChecked(sandbox, 'mkdir', ['-p', '--', this.runtimeRoot])
      // `mkdir -p` accepts a pre-existing symlink to a directory, so prove the
      // reserved path is a real directory before it holds adapter state.
      const probe = await sandbox.run('bash', {
        args: ['-c', 'test -d "$1" && ! test -L "$1"', 'dsh-tensorlake', this.runtimeRoot],
      })
      if (probe.exitCode !== 0) {
        throw new Error(`dsh-tensorlake: runtime root must be a real directory: ${this.runtimeRoot}`)
      }
      await runChecked(sandbox, 'chmod', ['700', '--', this.runtimeRoot])
      return sandbox
    } catch (error: unknown) {
      try {
        await sandbox.terminate()
      } catch {
        // TODO(tensorlake-setup-rollback): Add retry state only if a real double
        // failure outlives Tensorlake's configured sandbox timeout.
      }
      throw error
    }
  }
}

export default TensorlakeRuntime
