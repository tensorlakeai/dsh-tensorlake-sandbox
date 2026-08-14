/**
 * Tensorlake provider for the filesystem capability seam. Paths, contents, and
 * atomic staging files remain inside the shared remote sandbox; metadata and
 * publication run through argv-style control commands because the sandbox
 * proxy exposes only whole-file read/write/list/delete primitives.
 * @module @tensorlakeai/dsh-sandbox/filesystem
 */

import { randomUUID } from 'node:crypto'
import { posix } from 'node:path'
import {
  FsError,
  FsTargetKey,
} from '@deepseek-ai/dsh-fs'
import type {
  FsDirEntry,
  FsInfo,
  FsPathInfo,
  FsTarget,
  FsVersion,
} from '@deepseek-ai/dsh-fs'
import { isRemoteMissing, runChecked, TensorlakeCommandError } from '../runtime.ts'
import type { Sandbox } from '../runtime.ts'
import {
  decodeDiffBasis,
  decodeProviderText,
  GuardedFileSystem,
  mapProviderFsError,
  sortDirEntries,
} from './provider.ts'
import {
  DANGLING_SENTINEL,
  decodeCanonicalPath,
  parseListing,
  parseStatFields,
  STAT_FIELD_COUNT,
  STAT_FIELDS,
  versionOf,
} from './stat.ts'
import type { RemoteStat } from './stat.ts'

const BINARY_SAMPLE_BYTES = 8192
const SHELL_NAME = 'dsh-fs-tensorlake'

const CANONICAL_PATH_SCRIPT = 'set -o pipefail; realpath -mz -- "$1" | base64 -w0'

const LISTING_SCRIPT = [
  'set -o pipefail',
  'find "$1" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d \'\' dsh_entry; do',
  `  stat --printf '%n\\0${STAT_FIELDS}' -- "$dsh_entry" || exit 1`,
  '  dsh_stat_error=$(mktemp) || exit 1',
  `  if LC_ALL=C stat -L --printf '${STAT_FIELDS}' -- "$dsh_entry" 2>"$dsh_stat_error"; then`,
  '    :',
  '  elif test -L "$dsh_entry" && grep -q "No such file or directory" "$dsh_stat_error"; then',
  `    printf '${DANGLING_SENTINEL}\\0\\0\\0\\0\\0'`,
  '  else',
  '    cat "$dsh_stat_error" >&2',
  '    rm -f -- "$dsh_stat_error"',
  '    exit 1',
  '  fi',
  '  rm -f -- "$dsh_stat_error"',
  '  realpath -mz -- "$dsh_entry" || exit 1',
  'done | base64 -w0',
].join('\n')

const GUARDED_CREATE_SCRIPT
  = 'if ln -T -- "$1" "$2" 2>/dev/null; then printf created; elif test -e "$2" || test -L "$2"; then printf exists; else exit 1; fi'

function assertNotAborted(signal: AbortSignal | undefined, operation: string): void {
  if (signal?.aborted === true) throw new FsError(`${operation} aborted`, 'FS_ABORTED')
}

/** Classify sandbox-proxy misses plus control-command not-found stderr. */
function isPathMissing(error: unknown): boolean {
  return isRemoteMissing(error)
    || (error instanceof TensorlakeCommandError && /no such file or directory/i.test(error.result.stderr))
}

function mapError(error: unknown, operation: string, displayPath: string, signal?: AbortSignal): FsError {
  return mapProviderFsError(error, operation, displayPath, { signal, isNotFound: isPathMissing })
}

function infoType(kind: RemoteStat['kind']): FsInfo['type'] {
  return kind === 'file' || kind === 'directory' ? kind : 'other'
}

/** Remote filesystem backend sharing the sandbox owned by `ctx.tensorlake`. */
export class TensorlakeFileSystem extends GuardedFileSystem<RemoteStat> {
  static inject = ['tensorlake']


  override async resolve(path: string, opts?: { cwd?: string; signal?: AbortSignal }): Promise<FsTarget> {
    assertNotAborted(opts?.signal, 'resolve')
    const displayPath = this.displayPathFor(path, opts?.cwd)
    const targetKey = await this.canonicalTargetKey(displayPath).catch((error: unknown) => {
      throw mapError(error, 'resolve', displayPath, opts?.signal)
    })
    assertNotAborted(opts?.signal, 'resolve')
    return { targetKey: FsTargetKey(targetKey), displayPath }
  }

  override processPath(target: FsTarget): string {
    return String(target.targetKey)
  }

  override fileUrl(target: FsTarget): string {
    const path = this.processPath(target)
    if (!path.startsWith('/')) throw new Error(`fs-tensorlake: expected an absolute process path: ${JSON.stringify(path)}`)
    return `file://${path.split('/').map(encodeURIComponent).join('/')}`
  }

  override contains(parent: FsTarget, child: FsTarget): boolean {
    const parentPath = this.processPath(parent)
    const childPath = this.processPath(child)
    if (parentPath === childPath) return true
    const prefix = parentPath.endsWith('/') ? parentPath : `${parentPath}/`
    return childPath.startsWith(prefix)
  }

  override async stat(target: FsTarget, signal?: AbortSignal): Promise<FsInfo | undefined> {
    const entry = await this.probe(String(target.targetKey), target.displayPath, true, signal)
    if (entry === undefined) return undefined
    return {
      version: versionOf(String(target.targetKey), entry),
      type: infoType(entry.kind),
      ...(entry.kind === 'file' ? { size: entry.size } : {}),
    }
  }

  override async lstat(path: string, opts?: { cwd?: string }, signal?: AbortSignal): Promise<FsPathInfo | undefined> {
    assertNotAborted(signal, 'lstat')
    const displayPath = this.displayPathFor(path, opts?.cwd)
    const entry = await this.probe(displayPath, displayPath, false, signal)
    if (entry === undefined) return undefined
    return {
      version: versionOf(displayPath, entry),
      type: entry.kind,
      ...(entry.kind === 'file' ? { size: entry.size } : {}),
    }
  }

  override async readText(target: FsTarget, signal?: AbortSignal): Promise<string> {
    const bytes = await this.readRegular(target, 'read', signal)
    return decodeProviderText(bytes, target.displayPath, BINARY_SAMPLE_BYTES)
  }

  override async readBytes(target: FsTarget, signal: AbortSignal | undefined, maxBytes: number): Promise<Uint8Array> {
    const info = await this.requireRegular(target, signal)
    if (info.size !== undefined && info.size > maxBytes) {
      throw new FsError(`cannot read "${target.displayPath}": ${info.size} bytes exceeds the ${maxBytes}-byte limit`, 'FS_TOO_LARGE')
    }
    const bytes = await this.readAfterCheck(target, 'read', signal)
    // The proxy returns whole files, so a post-stat grower is caught on the
    // complete result rather than during transfer.
    if (bytes.length > maxBytes) {
      throw new FsError(`cannot read "${target.displayPath}": content exceeds the ${maxBytes}-byte limit`, 'FS_TOO_LARGE')
    }
    return bytes
  }

  override async streamText(target: FsTarget, signal?: AbortSignal): Promise<AsyncIterable<string>> {
    await this.requireRegular(target, signal)
    // The sandbox proxy has no ranged or streaming read, so each iteration is
    // one validated whole-file read delivered as a single chunk.
    const read = (): Promise<Uint8Array> => this.readAfterCheck(target, 'read', signal)
    const displayPath = target.displayPath
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<string> {
        const text = decodeProviderText(await read(), displayPath, BINARY_SAMPLE_BYTES)
        assertNotAborted(signal, 'read')
        if (text.length > 0) yield text
      },
    }
  }

  override async listDir(target: FsTarget, signal?: AbortSignal): Promise<FsDirEntry[]> {
    await this.requireDirectory(target, signal)
    try {
      const sandbox = await this.ctx.tensorlake.getSandbox()
      assertNotAborted(signal, 'list')
      const listed = await runChecked(sandbox, 'bash', ['-c', LISTING_SCRIPT, SHELL_NAME, String(target.targetKey)])
      assertNotAborted(signal, 'list')
      const entries = parseListing(listed.stdout).map((record): FsDirEntry => {
        const name = posix.basename(record.path)
        return {
          name,
          type: record.resolved === undefined ? 'other' : infoType(record.resolved.kind),
          target: { targetKey: FsTargetKey(record.canonicalPath), displayPath: posix.join(target.displayPath, name) },
          ...(record.resolved !== undefined ? { version: versionOf(record.canonicalPath, record.resolved) } : {}),
          ...(record.resolved?.kind === 'file' ? { size: record.resolved.size } : {}),
        }
      })
      return sortDirEntries(entries)
    } catch (error: unknown) {
      throw mapError(error, 'list', target.displayPath, signal)
    }
  }

  protected override async probeForMutation(
    target: FsTarget,
    signal: AbortSignal | undefined,
  ): Promise<{ facts: RemoteStat; version: FsVersion; regular: boolean } | undefined> {
    const entry = await this.probe(String(target.targetKey), target.displayPath, true, signal)
    if (entry === undefined) return undefined
    return {
      facts: entry,
      version: versionOf(String(target.targetKey), entry),
      regular: entry.kind === 'file',
    }
  }

  private displayPathFor(path: string, cwd: string | undefined): string {
    if (path.trim().length === 0) throw new FsError('file_path must be a non-empty string', 'FS_NOT_FOUND')
    return posix.resolve(cwd ?? this.ctx.tensorlake.cwd, path)
  }

  private async canonicalTargetKey(displayPath: string): Promise<string> {
    const sandbox = await this.ctx.tensorlake.getSandbox()
    return this.canonicalPath(sandbox, displayPath)
  }

  private async canonicalPath(sandbox: Sandbox, path: string): Promise<string> {
    try {
      const result = await runChecked(sandbox, 'bash', ['-c', CANONICAL_PATH_SCRIPT, SHELL_NAME, path])
      return decodeCanonicalPath(result.stdout)
    } catch (error: unknown) {
      if (error instanceof TensorlakeCommandError) throw new Error(error.result.stderr || error.message, { cause: error })
      throw error
    }
  }

  private async probe(
    path: string,
    displayPath: string,
    follow: boolean,
    signal?: AbortSignal,
  ): Promise<RemoteStat | undefined> {
    assertNotAborted(signal, 'stat')
    try {
      const sandbox = await this.ctx.tensorlake.getSandbox()
      // stat itself interprets the \0 escapes in the format argument.
      const result = await sandbox.run('stat', {
        args: [...(follow ? ['-L'] : []), '--printf', STAT_FIELDS, '--', path],
      })
      assertNotAborted(signal, 'stat')
      if (result.exitCode !== 0) {
        if (/no such file or directory/i.test(result.stderr)) return undefined
        throw new TensorlakeCommandError('stat', result)
      }
      const fields = result.stdout.split('\0')
      if (fields.at(-1) !== '') throw new Error('fs-tensorlake: stat transport is not NUL-terminated')
      return parseStatFields(fields.slice(0, STAT_FIELD_COUNT))
    } catch (error: unknown) {
      throw mapError(error, 'stat', displayPath, signal)
    }
  }

  private async readRegular(target: FsTarget, operation: string, signal?: AbortSignal): Promise<Uint8Array> {
    await this.requireRegular(target, signal)
    return this.readAfterCheck(target, operation, signal)
  }

  private async readAfterCheck(target: FsTarget, operation: string, signal?: AbortSignal): Promise<Uint8Array> {
    try {
      const sandbox = await this.ctx.tensorlake.getSandbox()
      const bytes = await sandbox.readFile(String(target.targetKey))
      assertNotAborted(signal, operation)
      return bytes
    } catch (error: unknown) {
      throw mapError(error, operation, target.displayPath, signal)
    }
  }

  protected override async readDiffBasis(target: FsTarget, signal?: AbortSignal): Promise<string | null> {
    return decodeDiffBasis(await this.readAfterCheck(target, 'read', signal), target.displayPath)
  }

  protected override async readForEdit(target: FsTarget, signal?: AbortSignal): Promise<string> {
    const bytes = await this.readAfterCheck(target, 'edit', signal)
    return decodeProviderText(bytes, target.displayPath, bytes.length)
  }

  protected override async writeAtomic(
    target: FsTarget,
    content: string,
    existing: RemoteStat | undefined,
    createIfAbsent: boolean,
    signal?: AbortSignal,
  ): Promise<FsVersion> {
    assertNotAborted(signal, 'write')
    const sandbox = await this.ctx.tensorlake.getSandbox()
    const targetPath = String(target.targetKey)
    const stagingDir = posix.join(posix.dirname(targetPath), `.dsh-${randomUUID()}.tmp`)
    const temporary = posix.join(stagingDir, 'content')
    let stagingDirCreated = false
    try {
      // A bare mkdir (never -p) proves this staging directory is private.
      await runChecked(sandbox, 'mkdir', ['--', stagingDir])
      stagingDirCreated = true
      await runChecked(sandbox, 'chmod', ['700', '--', stagingDir])
      assertNotAborted(signal, 'write')
      await sandbox.writeFile(temporary, new TextEncoder().encode(content))
      const mode = existing === undefined ? 0o600 : existing.mode & 0o777
      await runChecked(sandbox, 'chmod', [mode.toString(8), '--', temporary])
      assertNotAborted(signal, 'write')
      const staged = await this.probe(temporary, target.displayPath, true, signal)
      if (staged === undefined || staged.kind !== 'file') {
        throw new Error('fs-tensorlake: staged content disappeared before publication')
      }
      if (createIfAbsent) {
        const publication = await sandbox.run('bash', {
          args: ['-c', GUARDED_CREATE_SCRIPT, SHELL_NAME, temporary, targetPath],
        })
        if (publication.exitCode !== 0) throw new TensorlakeCommandError('ln', publication)
        if (publication.stdout === 'exists') {
          throw new FsError(
            `cannot overwrite existing "${target.displayPath}" without reading it first`,
            'FS_NOT_OBSERVED',
          )
        }
        if (publication.stdout !== 'created') {
          throw new Error('fs-tensorlake: guarded create returned an invalid publication result')
        }
      } else {
        await runChecked(sandbox, 'mv', ['-T', '--', temporary, targetPath])
      }
      try {
        await runChecked(sandbox, 'rm', ['-rf', '--', stagingDir])
      } catch {
        // The target is already committed; leftover private staging cannot turn that write into a failure.
      }
      // Publication preserves the staged inode (rename and hard link both keep
      // it), so the pre-publication facts are the committed version's facts.
      return versionOf(targetPath, staged)
    } catch (error: unknown) {
      if (stagingDirCreated) {
        try {
          await runChecked(sandbox, 'rm', ['-rf', '--', stagingDir])
        } catch {
          // Only the private staging directory is swallowed; the original failure owns the operation.
        }
      }
      throw mapError(error, 'write', target.displayPath, signal)
    }
  }
}

export default TensorlakeFileSystem
