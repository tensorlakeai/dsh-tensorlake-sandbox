/**
 * Tensorlake Service Provider for the subprocess capability seam. Each handle
 * starts through the shared sandbox daemon and retains exit-status and spill
 * paths in that remote world.
 * @module @tensorlakeai/dsh-sandbox/subprocess
 */

import { randomUUID } from 'node:crypto'
import { posix } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type {
  SubprocessHandle,
  SubprocessSpawnSpec,
  SubprocessTerminalHandle,
  SubprocessTerminalSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'
import { runChecked, TensorlakeCommandError } from '../runtime.ts'
import { TrackedSubprocessRuntime } from './provider.ts'
import { TensorlakeSubprocessHandle } from './process.ts'
import { SHELL_NAME } from './remote.ts'
import { spawnTensorlakeTerminal } from './terminal.ts'

/** Configuration for the Tensorlake subprocess adapter. */
export interface Config {
  /** Remote status/liveness poll cadence in milliseconds; each tick is one proxy request. */
  pollMs?: number
}

interface SchemaResolvedConfig extends Config {
  pollMs: number
}

/** Tensorlake process manager registered as `ctx.subprocess`. */
export class TensorlakeSubprocessRuntime extends TrackedSubprocessRuntime {
  static inject = ['tensorlake']

  static Config: z<Config> = z.object({
    pollMs: z.number().default(20),
  })

  private readonly pollMs: number

  /** Create the Tensorlake subprocess service and bind its disposal policy. */
  constructor(ctx: Context, config: Config) {
    super(ctx, 'subprocess-tensorlake', 'tensorlake subprocess teardown')
    // Schemastery fills pollMs before construction; the type does not encode that step.
    const { pollMs } = config as SchemaResolvedConfig
    if (!Number.isSafeInteger(pollMs) || pollMs <= 0) {
      throw new Error('subprocess-tensorlake: pollMs must be a positive safe integer')
    }
    this.pollMs = pollMs
  }

  /** @inheritdoc */
  async resolveExecutable(
    command: string,
    env?: Readonly<Record<string, string>>,
    signal?: AbortSignal,
  ): Promise<string> {
    if (command.length === 0) throw new Error('subprocess-tensorlake: executable name must be non-empty')
    signal?.throwIfAborted()
    const sandbox = await this.ctx.tensorlake.getSandbox()
    if (posix.isAbsolute(command)) {
      const probe = await sandbox.run('bash', {
        args: ['-c', 'test -f "$1" && test -x "$1"', SHELL_NAME, command],
      })
      signal?.throwIfAborted()
      if (probe.exitCode !== 0) {
        throw new Error(`subprocess-tensorlake: ${JSON.stringify(command)} is not an executable file`)
      }
      return command
    }
    if (command.includes('/')) {
      throw new Error(
        `subprocess-tensorlake: command ${JSON.stringify(command)} is a relative path; use an absolute path or a bare PATH name`,
      )
    }
    let result: Awaited<ReturnType<typeof runChecked>>
    try {
      result = await runChecked(sandbox, 'bash', ['-c', 'command -v -- "$1"', SHELL_NAME, command], {
        workingDir: this.ctx.tensorlake.cwd,
        // The daemon merges this override onto its ambient environment, so an
        // explicit PATH wins the lookup without rebuilding the whole set.
        ...(env?.PATH === undefined ? {} : { env: { PATH: env.PATH } }),
      })
    } catch (error: unknown) {
      if (error instanceof TensorlakeCommandError) {
        throw new Error(`subprocess-tensorlake: executable ${JSON.stringify(command)} was not found on PATH`, { cause: error })
      }
      throw error
    }
    signal?.throwIfAborted()
    const executable = result.stdout.trim()
    if (executable.includes('\n') || (!posix.isAbsolute(executable) && !executable.includes('/'))) {
      throw new Error(`subprocess-tensorlake: executable ${JSON.stringify(command)} did not resolve to one absolute path`)
    }
    // A relative result comes from a relative PATH entry; the lookup ran with the shared cwd.
    return posix.resolve(this.ctx.tensorlake.cwd, executable)
  }

  /** @inheritdoc */
  spawn(spec: SubprocessSpawnSpec): SubprocessHandle {
    this.guardSpawn(spec)
    const stateDir = posix.join(this.ctx.tensorlake.runtimeRoot, 'processes', randomUUID())
    return this.adoptHandle(new TensorlakeSubprocessHandle(this.ctx.tensorlake, spec, stateDir, this.pollMs))
  }

  /** @inheritdoc */
  async spawnTerminal(spec: SubprocessTerminalSpawnSpec): Promise<SubprocessTerminalHandle> {
    this.guardTerminalSpawn(spec)
    const stateDir = posix.join(this.ctx.tensorlake.runtimeRoot, 'terminals', randomUUID())
    return this.adoptTerminal(spec.signal, setupSignal => spawnTensorlakeTerminal(
      this.ctx.tensorlake,
      { ...spec, signal: setupSignal },
      stateDir,
      this.pollMs,
    ))
  }
}

export default TensorlakeSubprocessRuntime
