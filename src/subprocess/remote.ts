/**
 * Shared remote-control helpers for the Tensorlake subprocess adapter: poll
 * ticks, tolerant process-group signalling, and group/session liveness reads.
 * The sandbox daemon signals only single processes, so every tree-scoped
 * operation goes through one `bash` builtin `kill` on negative group ids.
 */

import { isRemoteMissing, TensorlakeCommandError } from '../runtime.ts'
import type { Sandbox } from '../runtime.ts'
import { parsePositiveId } from './provider.ts'

/** Shell `$0` used by every adapter-internal bash invocation. */
export const SHELL_NAME = 'dsh-subprocess-tensorlake'

/**
 * Signal remote process groups, tolerating the shared teardown outcomes: a
 * nonzero `kill` (groups already gone) and a disappeared sandbox. Both the
 * pgid-keyed process ladder and the sid-keyed terminal ladder deliver signals
 * through this single tolerance so they cannot drift apart.
 * @param sandbox - Live SDK handle.
 * @param groups - Positive process-group ids to signal.
 * @param signal - `TERM` or `KILL`.
 */
export async function signalRemoteGroups(
  sandbox: Sandbox,
  groups: readonly number[],
  signal: 'TERM' | 'KILL',
): Promise<void> {
  // TODO(tensorlake-group-signal): Prefer a daemon-native group signal if the
  // sandbox API adds one; a userspace precheck cannot close the numeric-PGID
  // reuse race.
  try {
    const targets = groups.map(group => `-${group}`)
    const result = await sandbox.run('bash', {
      args: ['-c', `kill -${signal} -- "$@"`, SHELL_NAME, ...targets],
    })
    if (result.exitCode !== 0) throw new TensorlakeCommandError('kill', result)
  } catch (error: unknown) {
    if (!(error instanceof TensorlakeCommandError) && !isRemoteMissing(error)) throw error
  }
}

/**
 * Report whether any live process remains in one process group.
 * @param sandbox - Live SDK handle.
 * @param processGroupId - Positive process-group id to probe.
 * @returns `true` while at least one non-zombie member remains; `false` after
 *   the group is empty or the sandbox itself is gone.
 */
export async function groupAlive(sandbox: Sandbox, processGroupId: number): Promise<boolean> {
  let result: { exitCode: number; stdout: string }
  try {
    result = await sandbox.run('bash', {
      args: [
        '-c',
        'set -o pipefail; ps -eo pgid=,stat= | awk -v group="$1" \'$1 == group && $2 !~ /^[ZXx]/ { live=1 } END { if (live) print "live" }\'',
        SHELL_NAME,
        String(processGroupId),
      ],
    })
  } catch (error: unknown) {
    if (isRemoteMissing(error)) return false
    throw error
  }
  if (result.exitCode !== 0) return false
  return result.stdout.trim() === 'live'
}

/**
 * List the live process groups of one POSIX session.
 * @param sandbox - Live SDK handle.
 * @param sessionId - Positive session id to enumerate.
 * @returns Unique positive group ids; empty after the session or sandbox is gone.
 */
export async function sessionProcessGroups(sandbox: Sandbox, sessionId: number): Promise<number[]> {
  let result: { exitCode: number; stdout: string }
  try {
    result = await sandbox.run('bash', {
      args: [
        '-c',
        'set -o pipefail; ps -eo sid=,pgid=,stat= | awk -v sid="$1" \'$1 == sid && $3 !~ /^[ZXx]/ { print $2 }\'',
        SHELL_NAME,
        String(sessionId),
      ],
    })
  } catch (error: unknown) {
    if (isRemoteMissing(error)) return []
    throw error
  }
  if (result.exitCode !== 0) return []
  const groups = new Set<number>()
  for (const raw of result.stdout.trim().split(/\s+/)) {
    if (raw.length === 0) continue
    const group = parsePositiveId(
      raw,
      `subprocess-tensorlake: invalid process group ${JSON.stringify(raw)} in terminal session ${sessionId}`,
    )
    if (group <= 1) {
      throw new Error(`subprocess-tensorlake: unsafe process group ${group} in terminal session ${sessionId}`)
    }
    groups.add(group)
  }
  return [...groups]
}
