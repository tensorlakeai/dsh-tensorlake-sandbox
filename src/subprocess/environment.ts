/**
 * Remote-environment handling for Tensorlake process and terminal launchers.
 * The sandbox daemon merges explicit entries onto its own ambient environment
 * and cannot remove entries, so every launch rebuilds a complete environment
 * with `env -i` from the scrubbed ambient base plus the spec's explicit layer.
 */

import { Buffer } from 'node:buffer'
import { SENSITIVE_ENV_PATTERN } from '@deepseek-ai/dsh-subprocess'
import { runChecked } from '../runtime.ts'
import type { Sandbox } from '../runtime.ts'
import { SHELL_NAME } from './remote.ts'

const BASE64_TEXT = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/

/**
 * Read the sandbox daemon's ambient environment through ASCII base64 so the
 * proxy's UTF-8 stdout decoding cannot silently corrupt values.
 * @param sandbox - Shared Tensorlake execution world.
 * @returns Name/value entries in ambient order.
 */
export async function readAmbientEnvironment(sandbox: Sandbox): Promise<Map<string, string>> {
  const result = await runChecked(sandbox, 'bash', ['-c', 'set -o pipefail; env -0 | base64 -w0', SHELL_NAME])
  const encoded = result.stdout.trim()
  if (!BASE64_TEXT.test(encoded)) {
    throw new Error('subprocess-tensorlake: ambient environment transport returned invalid base64')
  }
  let raw: string
  try {
    raw = new TextDecoder('utf-8', { fatal: true }).decode(Buffer.from(encoded, 'base64'))
  } catch (error: unknown) {
    throw new Error('subprocess-tensorlake: ambient environment is not valid UTF-8', { cause: error })
  }
  const environment = new Map<string, string>()
  for (const entry of raw.split('\0')) {
    if (entry.length === 0) continue
    const separator = entry.indexOf('=')
    if (separator <= 0) continue
    environment.set(entry.slice(0, separator), entry.slice(separator + 1))
  }
  return environment
}

/**
 * Build the complete `NAME=value` argument list for one `env -i` launch:
 * the ambient base minus harness-private and credential-shaped names, overlaid
 * with the spec's explicit entries (an `undefined` value is the seam's
 * tombstone and removes an ambient entry).
 * @param ambient - The daemon's complete ambient environment.
 * @param explicit - Deliberate caller overrides applied after the scrub.
 * @returns `env`-ready `NAME=value` words.
 */
export function environmentArguments(
  ambient: ReadonlyMap<string, string>,
  explicit: Readonly<NodeJS.ProcessEnv> | undefined,
): string[] {
  const environment = new Map<string, string>()
  for (const [name, value] of ambient) {
    if (name.toUpperCase().startsWith('DSH_') || SENSITIVE_ENV_PATTERN.test(name)) continue
    environment.set(name, value)
  }
  for (const [name, value] of Object.entries(explicit ?? {})) {
    if (name.length === 0 || name.includes('=') || name.includes('\0') || value?.includes('\0') === true) {
      throw new Error('subprocess-tensorlake: environment entries require non-empty NUL-free names without = and NUL-free values')
    }
    if (value === undefined) environment.delete(name)
    else environment.set(name, value)
  }
  return [...environment].map(([name, value]) => `${name}=${value}`)
}
