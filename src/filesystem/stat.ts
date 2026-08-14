/**
 * Remote metadata vocabulary for the Tensorlake filesystem adapter. The
 * sandbox proxy exposes no stat or rename primitives, so `stat`, `realpath`,
 * and `find` runs project every path fact the provider needs; the parsers here
 * validate those ASCII/NUL transports before any fact is trusted.
 */

import { createHash } from 'node:crypto'
import { Buffer } from 'node:buffer'
import { posix } from 'node:path'
import { FsVersion } from '@deepseek-ai/dsh-fs'

/** `stat --printf` fields consumed per path: type, size, raw mode, inode, ns mtime. */
export const STAT_FIELDS = '%F\\0%s\\0%f\\0%i\\0%y\\0'

/** How many NUL-terminated fields {@link STAT_FIELDS} emits. */
export const STAT_FIELD_COUNT = 5

/** Follow-block sentinel a listing emits for a dangling symbolic link. */
export const DANGLING_SENTINEL = '!dangling!'

/** Parsed facts of one remote path entry. */
export interface RemoteStat {
  /** Path-entry classification derived from `stat`'s `%F` file-type word. */
  kind: 'file' | 'directory' | 'symlink' | 'other'
  /** Byte size from `%s`. */
  size: number
  /** Raw `st_mode` bits from `%f`, including the file-type bits. */
  mode: number
  /** Inode number from `%i`, kept as its decimal text. */
  inode: string
  /** Nanosecond-precision modification time text from `%y`. */
  mtime: string
}

function kindOfFileType(fileType: string): RemoteStat['kind'] {
  switch (fileType) {
    case 'regular file':
    case 'regular empty file':
      return 'file'
    case 'directory':
      return 'directory'
    case 'symbolic link':
      return 'symlink'
    default:
      return 'other'
  }
}

/**
 * Parse one {@link STAT_FIELDS} group into validated facts.
 * @param fields - Exactly {@link STAT_FIELD_COUNT} decoded field strings.
 * @returns the parsed facts.
 * @throws when a numeric field is not what `stat` can legitimately print.
 */
export function parseStatFields(fields: readonly string[]): RemoteStat {
  const [fileType, rawSize, rawMode, inode, mtime] = fields
  if (fields.length !== STAT_FIELD_COUNT || fileType === undefined || rawSize === undefined
    || rawMode === undefined || inode === undefined || mtime === undefined) {
    throw new Error('fs-tensorlake: stat transport returned a malformed field group')
  }
  const size = Number(rawSize)
  const mode = Number.parseInt(rawMode, 16)
  if (!/^\d+$/.test(rawSize) || !Number.isSafeInteger(size)
    || !/^[0-9a-f]+$/.test(rawMode) || !Number.isSafeInteger(mode)
    || !/^\d+$/.test(inode) || mtime.length === 0) {
    throw new Error('fs-tensorlake: stat transport returned invalid numeric facts')
  }
  return { kind: kindOfFileType(fileType), size, mode, inode, mtime }
}

/**
 * Derive the opaque freshness token for one path entry. Every atomic write
 * publishes a freshly created inode, so the token changes on each mutation
 * even when content, size, and timestamps repeat.
 * @param canonicalPath - The identity the token is scoped to.
 * @param stat - Parsed facts of the entry.
 * @returns the branded version token.
 */
export function versionOf(canonicalPath: string, stat: RemoteStat): FsVersion {
  const facts = JSON.stringify([canonicalPath, stat.kind, stat.size, stat.mode, stat.inode, stat.mtime])
  return FsVersion(`tensorlake:${createHash('sha256').update(facts).digest('hex')}`)
}

/**
 * Decode `realpath -mz | base64 -w0` transport output into one canonical path.
 * @param encoded - Base64 text captured from the control command.
 * @returns the canonical absolute POSIX path.
 * @throws when framing, encoding, or absoluteness is violated.
 */
export function decodeCanonicalPath(encoded: string): string {
  const framed = decodeBase64(encoded, 'canonical path')
  if (framed.length < 2 || framed.at(-1) !== 0 || framed.subarray(0, -1).includes(0)) {
    throw new Error('fs-tensorlake: canonical path transport returned invalid NUL framing')
  }
  const path = decodeUtf8(framed.subarray(0, -1), 'canonical path')
  if (!posix.isAbsolute(path)) throw new Error('fs-tensorlake: canonical path is not absolute')
  return path
}

/** One parsed directory-listing record. */
export interface ListingRecord {
  /** Absolute remote path of the child as listed. */
  path: string
  /** Facts of the child itself, without following a symbolic link. */
  entry: RemoteStat
  /** Facts of the followed child, or `undefined` for a dangling link. */
  resolved: RemoteStat | undefined
  /** Canonical absolute path after resolving every link component. */
  canonicalPath: string
}

/** How many NUL-terminated fields each listing record carries. */
const LISTING_FIELD_COUNT = 1 + STAT_FIELD_COUNT + STAT_FIELD_COUNT + 1

/**
 * Decode one base64 listing transport into per-child records.
 * @param encoded - Base64 text captured from the listing command.
 * @returns one record per direct child, in traversal order.
 * @throws when the transport framing or any field group is invalid.
 */
export function parseListing(encoded: string): ListingRecord[] {
  if (encoded.length === 0) return []
  const text = decodeUtf8(decodeBase64(encoded, 'listing'), 'listing')
  const fields = text.split('\0')
  if (fields.at(-1) !== '') throw new Error('fs-tensorlake: listing transport is not NUL-terminated')
  fields.pop()
  if (fields.length % LISTING_FIELD_COUNT !== 0) {
    throw new Error('fs-tensorlake: listing transport returned a partial record')
  }
  const records: ListingRecord[] = []
  for (let start = 0; start < fields.length; start += LISTING_FIELD_COUNT) {
    const path = fields[start] as string
    const entry = parseStatFields(fields.slice(start + 1, start + 1 + STAT_FIELD_COUNT))
    const followFields = fields.slice(start + 1 + STAT_FIELD_COUNT, start + 1 + 2 * STAT_FIELD_COUNT)
    const resolved = followFields[0] === DANGLING_SENTINEL ? undefined : parseStatFields(followFields)
    const canonicalPath = fields[start + LISTING_FIELD_COUNT - 1] as string
    if (!posix.isAbsolute(path) || !posix.isAbsolute(canonicalPath)) {
      throw new Error('fs-tensorlake: listing transport returned a non-absolute path')
    }
    records.push({ path, entry, resolved, canonicalPath })
  }
  return records
}

function decodeBase64(encoded: string, transport: string): Buffer {
  const bytes = Buffer.from(encoded, 'base64')
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(encoded)
    || encoded.length === 0 || bytes.toString('base64') !== encoded) {
    throw new Error(`fs-tensorlake: ${transport} transport returned invalid base64`)
  }
  return bytes
}

function decodeUtf8(bytes: Uint8Array, transport: string): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch (error: unknown) {
    throw new Error(`fs-tensorlake: ${transport} transport is not valid UTF-8`, { cause: error })
  }
}
