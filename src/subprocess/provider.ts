/** Process-management mechanics kept private so the bundle targets the public dsh subprocess API. */

import { PassThrough } from 'node:stream'
import type { Context } from '@deepseek-ai/cordis'
import SubprocessRuntime from '@deepseek-ai/dsh-subprocess'
import type {
  SubprocessCollect,
  SubprocessCollectedOutputs,
  SubprocessHandle,
  SubprocessOutputMode,
  SubprocessOutputReader,
  SubprocessSpawnSpec,
  SubprocessStdio,
  SubprocessTerminalHandle,
  SubprocessTerminalSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'

export const MAX_TIMER_DELAY_MS = 2_147_483_647

/**
 * Normalize an unknown rejection into an Error.
 * @param error - Any thrown or rejected value.
 * @returns The value itself when already an Error, else a stringified wrapper.
 */
export function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

/**
 * Resolve after one duration.
 * @param ms - Milliseconds to wait.
 * @returns Settles after the timeout.
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Wait one poll interval or until the signal aborts.
 * @param pollMs - Poll cadence in milliseconds.
 * @param signal - Optional abort that ends the wait early.
 * @returns `true` after a full tick, `false` when aborted first.
 */
export function waitTick(pollMs: number, signal?: AbortSignal): Promise<boolean> {
  if (signal?.aborted === true) return Promise.resolve(false)
  return new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve(true)
    }, pollMs)
    const onAbort = (): void => {
      clearTimeout(timer)
      resolve(false)
    }
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * Parse one positive process/group/session id printed by `ps`.
 * @param value - Raw `ps` output for one id column.
 * @param message - Error message when the output is not one positive id.
 * @returns The parsed id.
 */
export function parsePositiveId(value: string, message: string): number {
  const raw = value.trim()
  const id = Number(raw)
  if (!/^[1-9][0-9]*$/.test(raw) || !Number.isSafeInteger(id)) throw new Error(message)
  return id
}

/**
 * Enforce the seam's documented grace bound (positive, finite, one Node
 * timer); an unbounded grace would make a provider's force-escalation
 * deadline unreachable.
 * @param graceMs - The spec's cleanup grace in milliseconds.
 */
export function requireRepresentableGrace(graceMs: number): void {
  if (!Number.isFinite(graceMs) || graceMs <= 0 || graceMs > MAX_TIMER_DELAY_MS) {
    throw new Error(`subprocess graceMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`)
  }
}

/**
 * Report whether one output disposition buffers boundedly in the provider.
 * @param mode - A spec's stdout or stderr disposition.
 * @returns `true` for the collect object form.
 */
export function isCollectMode(mode: SubprocessOutputMode): mode is SubprocessCollect {
  return mode !== 'pipe' && mode !== 'inherit'
}

/**
 * Report whether one output disposition also spills the complete stream.
 * @param mode - A spec's stdout or stderr disposition.
 * @returns `true` for a collect object carrying a spill cap.
 */
export function hasSpillMode(mode: SubprocessOutputMode): mode is SubprocessCollect & { spill: { maxBytes: number } } {
  return isCollectMode(mode) && mode.spill !== undefined
}

interface TerminalSetup {
  done: Promise<void>
  controller: AbortController
}

/**
 * Subprocess service base that owns live-handle and terminal registration,
 * setup-cancellation bookkeeping, and the disposal ladder: reject new starts,
 * abort and join in-flight terminal setups, terminate and join every retained
 * handle and terminal, and aggregate cleanup failures. Subclasses own
 * substrate validation, spawning, and every remote mechanism.
 */
export abstract class TrackedSubprocessRuntime extends SubprocessRuntime {
  private readonly live = new Set<SubprocessHandle>()
  private readonly terminals = new Set<SubprocessTerminalHandle>()
  private readonly terminalSetups = new Set<TerminalSetup>()
  private disposingFlag = false

  /**
   * Register the service and bind its disposal ladder.
   * @param ctx - Cordis context receiving the service.
   * @param providerName - Message prefix naming the concrete provider.
   * @param teardownLabel - Effect label for the disposal ladder.
   */
  protected constructor(
    ctx: Context,
    private readonly providerName: string,
    teardownLabel: string,
  ) {
    super(ctx)
    ctx.effect(() => async () => {
      this.disposingFlag = true
      for (const setup of this.terminalSetups) {
        setup.controller.abort(new Error(`${this.providerName}: service disposed during terminal setup`))
      }
      await Promise.all([...this.terminalSetups].map(setup => setup.done))
      const handles = [...this.live]
      const terminals = [...this.terminals]
      const pending: Promise<unknown>[] = []
      for (const handle of handles) {
        handle.terminate()
        pending.push(handle.waitForExit().then(async () => {
          await handle.done.catch(() => undefined)
          this.live.delete(handle)
        }))
      }
      for (const terminal of terminals) {
        pending.push(terminal.terminate().then(() => { this.terminals.delete(terminal) }))
      }
      const outcomes = await Promise.allSettled(pending)
      const failures = outcomes.flatMap<unknown>(outcome => outcome.status === 'rejected'
        ? [outcome.reason as unknown]
        : [])
      if (failures.length === 1) throw asError(failures[0])
      if (failures.length > 1) throw new AggregateError(failures, `${this.providerName}: teardown failed`)
    }, teardownLabel)
  }

  /** Whether disposal has begun; new starts must be rejected. */
  protected get disposing(): boolean {
    return this.disposingFlag
  }

  /**
   * Validate one spawn request against the seam bounds shared by providers.
   * @param spec - The fully resolved spawn request.
   */
  protected guardSpawn(spec: SubprocessSpawnSpec): void {
    if (this.disposingFlag) throw new Error(`${this.providerName}: service is disposing`)
    const program = spec.argv[0]
    if (program === undefined || program.length === 0) {
      throw new Error('invalid argv: expected a non-empty program name at argv[0]')
    }
    requireRepresentableGrace(spec.graceMs)
    if (spec.signal?.aborted === true) {
      throw new Error(`aborted before spawn: ${String(spec.signal.reason)}`)
    }
  }

  /**
   * Validate one terminal spawn request against the shared seam bounds.
   * @param spec - The fully resolved terminal request.
   */
  protected guardTerminalSpawn(spec: SubprocessTerminalSpawnSpec): void {
    if (this.disposingFlag) throw new Error(`${this.providerName}: service is disposing`)
    const program = spec.argv[0]
    if (program === undefined || program.length === 0) {
      throw new Error(`${this.providerName}: terminal argv must contain a program`)
    }
    requireRepresentableGrace(spec.graceMs)
    spec.signal?.throwIfAborted()
  }

  /**
   * Retain one live handle until its whole tree exits; disposal retries the
   * cleanup transaction of a handle whose automatic release failed.
   * @param handle - The just-spawned live handle.
   * @returns The same handle.
   */
  protected adoptHandle<H extends SubprocessHandle>(handle: H): H {
    this.live.add(handle)
    const release = async (): Promise<void> => {
      await handle.waitForExit()
      this.live.delete(handle)
    }
    void handle.done.then(release, release).catch((_automaticReleaseFailure: unknown) => {
      // Retain the handle so service disposal can retry its cleanup transaction.
    })
    return handle
  }

  /**
   * Run one cancellable terminal setup and retain the published terminal
   * until it exits or disposal tears it down.
   * @param signal - The spec's optional allocation-cancellation signal.
   * @param spawn - Allocates the terminal under the joined setup signal.
   * @returns The live terminal handle.
   */
  protected async adoptTerminal(
    signal: AbortSignal | undefined,
    spawn: (setupSignal: AbortSignal) => Promise<SubprocessTerminalHandle>,
  ): Promise<SubprocessTerminalHandle> {
    const done = Promise.withResolvers<void>()
    const setup: TerminalSetup = { done: done.promise, controller: new AbortController() }
    const setupSignal = signal === undefined
      ? setup.controller.signal
      : AbortSignal.any([signal, setup.controller.signal])
    this.terminalSetups.add(setup)
    try {
      const terminal = await spawn(setupSignal)
      this.terminals.add(terminal)
      // Allocation yields, so disposal can begin between the guard and here.
      if (this.disposingFlag) {
        await terminal.terminate()
        this.terminals.delete(terminal)
        throw new Error(`${this.providerName}: service disposed during terminal setup`)
      }
      const release = async (): Promise<void> => {
        await terminal.terminate()
        this.terminals.delete(terminal)
      }
      void terminal.done.then(release, release).catch((_automaticReleaseFailure: unknown) => {
        // Retain the terminal so service disposal can retry its cleanup transaction.
      })
      return terminal
    } finally {
      this.terminalSetups.delete(setup)
      done.resolve()
    }
  }
}

/**
 * Build the per-stream transports one provider handle publishes: a raw pipe
 * for each `'pipe'` disposition, a provider reader for each collect
 * disposition, and the `collected` map naming only the collect streams.
 * `'inherit'` streams get neither, since the child writes the host's
 * descriptor directly.
 * @param stdio - The spec's three stdio dispositions.
 * @param makeReader - Builds this provider's reader for one collect stream.
 * @returns The pipes, the per-stream readers, and the published `collected` map.
 */
export function buildProviderStdio<Reader extends SubprocessOutputReader>(
  stdio: SubprocessStdio,
  makeReader: (mode: SubprocessCollect, stream: 'stdout' | 'stderr') => Reader,
): {
  stdout: PassThrough | undefined
  stderr: PassThrough | undefined
  stdoutReader: Reader | undefined
  stderrReader: Reader | undefined
  collected: SubprocessCollectedOutputs
} {
  const stdout = stdio.stdout === 'pipe' ? new PassThrough() : undefined
  const stderr = stdio.stderr === 'pipe' ? new PassThrough() : undefined
  const stdoutReader = isCollectMode(stdio.stdout) ? makeReader(stdio.stdout, 'stdout') : undefined
  const stderrReader = isCollectMode(stdio.stderr) ? makeReader(stdio.stderr, 'stderr') : undefined
  return {
    stdout,
    stderr,
    stdoutReader,
    stderrReader,
    collected: {
      ...(stdoutReader !== undefined ? { stdout: stdoutReader } : {}),
      ...(stderrReader !== undefined ? { stderr: stderrReader } : {}),
    },
  }
}

/**
 * Termination bookkeeping for one managed process tree. The seam makes
 * {@link SubprocessHandle.terminate} fire-and-forget and
 * {@link SubprocessHandle.waitForExit} the place a failed termination surfaces,
 * so a provider needs exactly one transaction in flight, one recorded failure
 * that proven quiescence clears, and the guarantee that a late failure never
 * overwrites proven quiescence.
 */
export class ProviderTermination {
  private quiescent = false
  private attempt: Promise<void> | undefined
  private failure: Error | undefined

  /** Whether the whole tree has been observed gone; further termination is pointless. */
  get proven(): boolean {
    return this.quiescent
  }

  /** The in-flight termination transaction, or `undefined` when none is running. */
  get pending(): Promise<void> | undefined {
    return this.attempt
  }

  /** Record proven quiescence, discarding any failure a superseded attempt left. */
  markQuiescent(): void {
    this.quiescent = true
    this.failure = undefined
  }

  /**
   * Record a failure raised outside {@link begin}'s transaction, so the next
   * {@link throwFailure} surfaces it.
   * @param error - The cleanup failure to retain.
   */
  recordFailure(error: unknown): void {
    this.failure = asError(error)
  }

  /**
   * Rethrow the recorded termination failure so a waiter observes it instead
   * of waiting on a tree nothing is still terminating.
   */
  throwFailure(): void {
    if (this.failure !== undefined) throw this.failure
  }

  /**
   * Start one termination transaction, clearing the previous failure and
   * retaining a new one only while quiescence remains unproven.
   * @param run - The provider's remote termination ladder.
   */
  begin(run: () => Promise<void>): void {
    this.failure = undefined
    const attempt = run()
    this.attempt = attempt
    void attempt.then(
      () => { this.attempt = undefined },
      (error: unknown) => {
        if (!this.quiescent) this.failure = asError(error)
        this.attempt = undefined
      },
    )
  }

  /**
   * Poll one provider's liveness probe until the whole tree is gone, then
   * record quiescence. A termination failure recorded while waiting surfaces
   * to the waiter instead of leaving it polling a tree nothing is terminating.
   * @param alive - The provider's whole-tree liveness probe.
   * @param pollMs - Poll cadence between probes.
   * @param signal - The waiter's cancellation.
   * @returns `true` once the tree is proven gone, `false` when the waiter aborted first.
   */
  async awaitQuiescence(alive: () => Promise<boolean>, pollMs: number, signal?: AbortSignal): Promise<boolean> {
    while (await alive()) {
      this.throwFailure()
      if (!await waitTick(pollMs, signal)) return false
    }
    this.throwFailure()
    if (signal?.aborted === true) return false
    this.markQuiescent()
    return true
  }

  /**
   * Prove quiescence before reporting a monitoring failure, so a published
   * tree never outlives the handle that failed to watch it.
   * @param error - The original monitoring failure.
   * @param published - Whether a remote tree was ever published for this handle.
   * @param handle - The handle whose termination ladder proves quiescence.
   * @param message - Aggregate message when the rollback itself fails.
   * @returns The original failure, or an {@link AggregateError} joining it with the rollback failure.
   */
  async rollback(
    error: unknown,
    published: boolean,
    handle: Pick<SubprocessHandle, 'terminate' | 'waitForExit'>,
    message: string,
  ): Promise<unknown> {
    if (!published || this.quiescent) return error
    handle.terminate()
    try {
      await handle.waitForExit()
      return error
    } catch (cleanupError: unknown) {
      return new AggregateError([asError(error), asError(cleanupError)], message)
    }
  }
}

/**
 * Operation bookkeeping for one live terminal. The seam requires every
 * terminal operation to be refused once termination begins, every in-flight
 * operation to settle before cleanup runs, and {@link
 * SubprocessTerminalHandle.terminate} to be idempotent while its transaction
 * succeeds — a failed cleanup must stay retryable.
 */
export class TerminalOperationGate {
  private readonly controller = new AbortController()
  private readonly operations = new Set<Promise<unknown>>()
  private closing: Promise<void> | undefined

  /**
   * @param terminatingMessage - Provider-named reason for the abort and for refused operations.
   */
  constructor(private readonly terminatingMessage: string) {}

  /**
   * Run one terminal operation under the gate's cancellation, unless
   * termination already began.
   * @param operation - The operation, receiving the gate's abort signal.
   * @returns The operation's result.
   */
  run<T>(operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.controller.signal.aborted) return Promise.reject(new Error(this.terminatingMessage))
    const pending = operation(this.controller.signal)
    this.operations.add(pending)
    void pending.then(
      () => { this.operations.delete(pending) },
      () => { this.operations.delete(pending) },
    )
    return pending
  }

  /**
   * Join the one termination transaction, starting it on the first call:
   * refuse new operations, let in-flight ones settle, then run cleanup.
   * A rejected transaction is forgotten so a caller can retry it.
   * @param cleanup - The provider's terminal-session teardown.
   * @returns The shared transaction.
   */
  close(cleanup: () => Promise<void>): Promise<void> {
    if (this.closing !== undefined) return this.closing
    this.controller.abort(new Error(this.terminatingMessage))
    const closing = (async () => {
      await Promise.allSettled(this.operations)
      await cleanup()
    })()
    this.closing = closing
    void closing.catch((_cleanupFailure: unknown) => {
      this.closing = undefined
    })
    return closing
  }
}

export default SubprocessRuntime
