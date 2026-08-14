/** Provider mechanics kept private so the bundle targets the public dsh filesystem API. */

import FileSystem, { FsError } from '@deepseek-ai/dsh-fs'
import type {
  FsDirEntry,
  FsEditOutcome,
  FsEditRequest,
  FsInfo,
  FsTarget,
  FsVersion,
  FsWriteIntent,
  FsWriteOutcome,
} from '@deepseek-ai/dsh-fs'

export abstract class GuardedFileSystem<Existing> extends FileSystem {
  private readonly locks = new TargetMutationLocks()

  /** Require an existing directory before listing it. */
  protected async requireDirectory(target: FsTarget, signal?: AbortSignal): Promise<void> {
    const info = await this.stat(target, signal)
    if (info === undefined) throw new FsError(`cannot list "${target.displayPath}": not found`, 'FS_NOT_FOUND')
    if (info.type !== 'directory') {
      throw new FsError(`cannot list "${target.displayPath}": not a directory`, 'FS_NOT_DIRECTORY')
    }
  }

  /** Require an existing regular file before reading it. */
  protected async requireRegular(target: FsTarget, signal?: AbortSignal): Promise<FsInfo> {
    const info = await this.stat(target, signal)
    if (info === undefined) throw new FsError(`cannot read "${target.displayPath}": not found`, 'FS_NOT_FOUND')
    if (info.type !== 'file') {
      throw new FsError(`cannot read "${target.displayPath}": not a regular file`, 'FS_NOT_REGULAR_FILE')
    }
    return info
  }

  /**
   * Report the current facts of one mutation target.
   * @param target - The resolved target about to be mutated.
   * @param signal - Aborts the metadata round-trip.
   * @returns The substrate facts with the target's version and regular-file
   *   flag, or `undefined` when the target is absent.
   */
  protected abstract probeForMutation(
    target: FsTarget,
    signal: AbortSignal | undefined,
  ): Promise<{ facts: Existing; version: FsVersion; regular: boolean } | undefined>

  /**
   * Read the prior content a write reports as its diff basis.
   * @param target - The resolved target being overwritten.
   * @param signal - Aborts the read.
   * @returns The LF-normalized prior text, or `null` when it is not text.
   */
  protected abstract readDiffBasis(target: FsTarget, signal?: AbortSignal): Promise<string | null>

  /**
   * Read the raw storage text an edit matches against, in the file's own line-ending convention.
   * @param target - The resolved target being edited.
   * @param signal - Aborts the read.
   * @returns The current raw content.
   */
  protected abstract readForEdit(target: FsTarget, signal?: AbortSignal): Promise<string>

  /**
   * Publish new content atomically over the probed facts.
   * @param target - The resolved target to publish.
   * @param content - The exact storage text to publish.
   * @param existing - The probed facts, or `undefined` for a target that was absent.
   * @param createIfAbsent - Whether publication must fail if the target appeared meanwhile.
   * @param signal - Aborts before publication takes effect.
   * @returns The version the publication produced.
   */
  protected abstract writeAtomic(
    target: FsTarget,
    content: string,
    existing: Existing | undefined,
    createIfAbsent: boolean,
    signal?: AbortSignal,
  ): Promise<FsVersion>

  /** @inheritdoc */
  override writeText(
    target: FsTarget,
    content: string,
    expected?: FsWriteIntent,
    signal?: AbortSignal,
  ): Promise<FsWriteOutcome> {
    return this.guardedWriteTextWithSignal(target, content, expected, signal)
  }

  /** @inheritdoc */
  override editText(
    target: FsTarget,
    edit: FsEditRequest,
    expected?: { version: FsVersion },
    signal?: AbortSignal,
  ): Promise<FsEditOutcome> {
    return this.guardedEditTextWithSignal(target, edit, expected, signal)
  }

  private guardedWriteTextWithSignal(
    target: FsTarget,
    content: string,
    expected: FsWriteIntent | undefined,
    signal: AbortSignal | undefined,
  ): Promise<FsWriteOutcome> {
    return this.locks.run(String(target.targetKey), async () => {
      const probed = await this.probeForMutation(target, signal)
      if (probed !== undefined && !probed.regular) {
        throw new FsError(`cannot write "${target.displayPath}": not a regular file`, 'FS_NOT_REGULAR_FILE')
      }
      assertWriteIntent(probed?.version, expected, target.displayPath)
      const before = probed === undefined ? null : await this.readDiffBasis(target, signal)
      const version = await this.writeAtomic(target, content, probed?.facts, expected?.kind === 'createIfAbsent', signal)
      return {
        operation: probed === undefined ? 'create' : 'update',
        version,
        before,
        after: normalizeLineEndings(content),
      }
    })
  }

  private guardedEditTextWithSignal(
    target: FsTarget,
    edit: FsEditRequest,
    expected: { version: FsVersion } | undefined,
    signal: AbortSignal | undefined,
  ): Promise<FsEditOutcome> {
    return this.locks.run(String(target.targetKey), async () => {
      const probed = await this.probeForMutation(target, signal)
      if (probed === undefined) {
        throw new FsError(`cannot edit "${target.displayPath}": file changed since it was read`, 'FS_STALE_VERSION')
      }
      if (!probed.regular) {
        throw new FsError(`cannot edit "${target.displayPath}": not a regular file`, 'FS_NOT_REGULAR_FILE')
      }
      if (expected !== undefined && probed.version !== expected.version) {
        throw new FsError(`cannot edit "${target.displayPath}": file changed since it was read`, 'FS_STALE_VERSION')
      }
      const raw = await this.readForEdit(target, signal)
      const before = normalizeLineEndings(raw)
      const after = literalEdit(before, edit, target.displayPath)
      const storage = restoreLineEndings(after, detectsCrlf(raw))
      const version = await this.writeAtomic(target, storage, probed.facts, false, signal)
      return { version, before, after }
    })
  }
}

/**
 * Normalize CRLF sequences to LF for matching and diff bases.
 * @param value - Raw text.
 * @returns The LF-normalized text.
 */
export function normalizeLineEndings(value: string): string {
  return value.replaceAll('\r\n', '\n')
}

/**
 * Report whether a text's leading sample is dominated by CRLF line endings.
 * @param value - Raw storage text.
 * @returns `true` when CRLF sequences outnumber bare LF in the first 4096 characters.
 */
export function detectsCrlf(value: string): boolean {
  const sample = value.slice(0, 4096)
  const crlf = sample.split('\r\n').length - 1
  const lf = sample.split('\n').length - 1 - crlf
  return crlf > lf
}

/**
 * Restore a file's dominant line-ending convention on LF-normalized text.
 * @param value - LF-normalized text to store.
 * @param crlf - Whether the prior storage was CRLF-dominant.
 * @returns Text in the storage convention.
 */
export function restoreLineEndings(value: string, crlf: boolean): string {
  return crlf ? normalizeLineEndings(value).replaceAll('\n', '\r\n') : value
}

/**
 * Decode one whole regular file as UTF-8 text with the seam's binary
 * rejection: a NUL byte within the leading sample refuses the file.
 * @param bytes - Complete raw content.
 * @param displayPath - Path used in error messages.
 * @param binarySampleBytes - How many leading bytes the NUL probe inspects.
 * @returns The decoded text.
 * @throws {FsError} `FS_NOT_TEXT` for binary or invalid UTF-8 content.
 */
export function decodeProviderText(bytes: Uint8Array, displayPath: string, binarySampleBytes: number): string {
  if (bytes.subarray(0, binarySampleBytes).includes(0)) {
    throw new FsError(`cannot read "${displayPath}": binary file`, 'FS_NOT_TEXT')
  }
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error: unknown) {
    throw new FsError(`cannot read "${displayPath}": invalid UTF-8 text`, 'FS_NOT_TEXT', { cause: error })
  }
}

/**
 * Order one directory listing by the seam's stable name order, so repeated
 * listings of unchanged content compare equal whatever order the substrate
 * enumerated them in.
 * @param entries - The listing in substrate order; ordered in place.
 * @returns The same entries in stable name order.
 */
export function sortDirEntries(entries: FsDirEntry[]): FsDirEntry[] {
  return entries.sort((left, right) => left.name.localeCompare(right.name))
}

/**
 * Decode one file's prior content as the diff basis of a write. A basis that
 * is not text yields no basis rather than failing the write the caller
 * already authorized.
 * @param bytes - Complete raw prior content.
 * @param displayPath - Path used in error messages.
 * @returns The LF-normalized prior text, or `null` when the prior content is not text.
 */
export function decodeDiffBasis(bytes: Uint8Array, displayPath: string): string | null {
  try {
    return normalizeLineEndings(decodeProviderText(bytes, displayPath, bytes.length))
  } catch (error: unknown) {
    if (error instanceof FsError && error.code === 'FS_NOT_TEXT') return null
    throw error
  }
}

/**
 * Apply one literal-replacement edit to LF-normalized content with the seam's
 * exact-match rules.
 * @param content - LF-normalized current content.
 * @param request - The literal search/replace request.
 * @param displayPath - Path used in error messages.
 * @returns The edited content.
 * @throws {FsError} `FS_EDIT_NOT_FOUND` for an empty or unmatched needle and
 *   `FS_AMBIGUOUS_EDIT` when one replacement matches more than once.
 */
export function literalEdit(content: string, request: FsEditRequest, displayPath: string): string {
  const oldString = normalizeLineEndings(request.oldString)
  const newString = normalizeLineEndings(request.newString)
  if (oldString.length === 0) {
    throw new FsError(`cannot edit "${displayPath}": old_string must be non-empty`, 'FS_EDIT_NOT_FOUND')
  }
  const segments = content.split(oldString)
  if (segments.length === 1) {
    throw new FsError(`cannot edit "${displayPath}": old_string was not found`, 'FS_EDIT_NOT_FOUND')
  }
  if (request.replaceAll) return segments.join(newString)
  if (segments.length !== 2) {
    throw new FsError(`cannot edit "${displayPath}": old_string matched ${segments.length - 1} times`, 'FS_AMBIGUOUS_EDIT')
  }
  return `${segments[0]}${newString}${segments[1]}`
}

/**
 * Enforce a guarded write intent against the currently observed version.
 * @param existingVersion - The target's current version, or `undefined` when absent.
 * @param expected - The caller's write intent; omit for unconditional writes.
 * @param displayPath - Path used in error messages.
 * @throws {FsError} `FS_NOT_OBSERVED` for a guarded create over an existing
 *   target and `FS_STALE_VERSION` for a guarded replace of absent or changed content.
 */
export function assertWriteIntent(
  existingVersion: FsVersion | undefined,
  expected: FsWriteIntent | undefined,
  displayPath: string,
): void {
  if (expected?.kind === 'createIfAbsent' && existingVersion !== undefined) {
    throw new FsError(`cannot overwrite existing "${displayPath}" without reading it first`, 'FS_NOT_OBSERVED')
  }
  if (expected?.kind === 'replaceIfVersion') {
    if (existingVersion === undefined || existingVersion !== expected.version) {
      throw new FsError(`cannot write "${displayPath}": file changed since it was read`, 'FS_STALE_VERSION')
    }
  }
}

/**
 * Map one provider failure onto the seam's typed error vocabulary. Providers
 * supply their substrate's not-found classification; abort and permission
 * classification are shared.
 * @param error - The raw failure.
 * @param operation - Verb used in the message.
 * @param displayPath - Path used in the message.
 * @param options - The caller's abort signal and not-found classifier.
 * @returns The mapped typed error (an existing {@link FsError} passes through).
 */
export function mapProviderFsError(
  error: unknown,
  operation: string,
  displayPath: string,
  options: { signal?: AbortSignal | undefined; isNotFound: (error: unknown) => boolean },
): FsError {
  if (error instanceof FsError) return error
  if (options.signal?.aborted === true || (error instanceof DOMException && error.name === 'AbortError')) {
    return new FsError(`${operation} aborted`, 'FS_ABORTED', { cause: error })
  }
  if (options.isNotFound(error)) {
    return new FsError(`cannot ${operation} "${displayPath}": not found`, 'FS_NOT_FOUND', { cause: error })
  }
  if (/permission denied|operation not permitted/i.test(String(error))) {
    return new FsError(`cannot ${operation} "${displayPath}": permission denied`, 'FS_PERMISSION_DENIED', { cause: error })
  }
  return new FsError(`cannot ${operation} "${displayPath}": ${String(error)}`, 'FS_IO_ERROR', { cause: error })
}

/**
 * Serialize mutations per canonical target within the host process, so one
 * provider's read-check-publish transactions on the same target never
 * interleave.
 */
export class TargetMutationLocks {
  private readonly locks = new Map<string, Promise<void>>()

  /**
   * Run one mutation after every earlier mutation of the same target settles.
   * @param targetKey - The canonical target identity to serialize on.
   * @param operation - The mutation transaction.
   * @returns The operation's result.
   */
  async run<T>(targetKey: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.locks.get(targetKey) ?? Promise.resolve()
    const run = previous.then(operation, operation)
    const tail = run.then(() => undefined, () => undefined)
    this.locks.set(targetKey, tail)
    try {
      return await run
    } finally {
      if (this.locks.get(targetKey) === tail) this.locks.delete(targetKey)
    }
  }
}

export default FileSystem
