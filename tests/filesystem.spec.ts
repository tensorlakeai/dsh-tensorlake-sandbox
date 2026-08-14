import { Buffer } from 'node:buffer'
import { posix } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { FsTargetKey, FsVersion } from '@deepseek-ai/dsh-fs'
import TensorlakeFileSystem from '../src/filesystem/index.ts'
import { RemoteAPIError } from '../src/runtime.ts'
import type { CommandResult, RunOptions, Sandbox } from '../src/runtime.ts'
import type TensorlakeRuntime from '../src/runtime.ts'
import { describe, expect, it } from 'vitest'

/** The literal `stat --printf` format the provider sends; `stat` expands the escapes remotely. */
const STAT_FORMAT = String.raw`%F\0%s\0%f\0%i\0%y\0`

type NodeKind = 'file' | 'directory' | 'symlink' | 'other'

interface RemoteNode {
  kind: NodeKind
  bytes: Uint8Array
  mode: number
  inode: string
  mtime: string
  symlinkTarget?: string
}

const TYPE_BITS: Record<NodeKind, number> = {
  file: 0o100000,
  directory: 0o040000,
  symlink: 0o120000,
  other: 0o010000,
}

const FILE_TYPE_WORDS: Record<NodeKind, string> = {
  file: 'regular file',
  directory: 'directory',
  symlink: 'symbolic link',
  other: 'fifo',
}

/** Follow-block the listing script prints for a dangling link: the sentinel plus four empty fields. */
const DANGLING_FOLLOW = nulFields('!dangling!', '', '', '', '')

function bytes(value: string | readonly number[]): Uint8Array {
  return typeof value === 'string' ? new TextEncoder().encode(value) : Uint8Array.from(value)
}

function nulFields(...fields: readonly string[]): string {
  return fields.map(field => `${field}\0`).join('')
}

function encode(payload: string | readonly number[]): string {
  return Buffer.from(bytes(payload)).toString('base64')
}

/** One 12-field listing record: entry path, its own facts, its followed facts, and its canonical path. */
function listingRecord(path: string, canonicalPath: string, size = '1'): string {
  const group = nulFields('regular file', size, '81a4', '12', '2026-08-13 12:00:00.000000001 +0000')
  return `${nulFields(path)}${group}${group}${nulFields(canonicalPath)}`
}

function ok(stdout = ''): CommandResult {
  return { exitCode: 0, stdout, stderr: '' }
}

function notFound(command: string, path: string): CommandResult {
  return { exitCode: 1, stdout: '', stderr: `${command}: cannot stat '${path}': No such file or directory` }
}

/**
 * In-memory sandbox emulating the exact argv commands and whole-file proxy
 * primitives the provider issues: GNU `stat`, `realpath`, `find`, `mkdir`,
 * `chmod`, `mv -T`, `rm -rf`, the guarded `ln -T` publication, and
 * `readFile`/`writeFile`. Publication preserves the staged node identity, so
 * inode-derived versions behave as they do remotely.
 */
class FakeSandbox {
  readonly nodes = new Map<string, RemoteNode>()
  readonly commands: Array<{ command: string; args: readonly string[] }> = []
  readonly reads: string[] = []
  readonly writes: Array<{ path: string; text: string }> = []
  /** One-shot per-argv-command failure: a nonzero result or a thrown transport error. */
  readonly failures = new Map<string, CommandResult | Error>()
  canonicalOutput: string | undefined
  statOutput: string | undefined
  listingOutput: string | undefined
  publicationOutput: CommandResult | undefined
  stagedProbe: 'missing' | 'directory' | undefined
  nextReadError: unknown
  nextWriteError: unknown
  grownRead: Uint8Array | undefined
  abortOnRead: AbortController | undefined
  abortOnPublish: AbortController | undefined
  competitor: { path: string; kind: 'file' | 'directory'; text: string } | undefined
  private clock = 0

  constructor() {
    this.dir('/')
    this.dir('/workspace')
  }

  dir(path: string, mode = 0o755): void {
    this.nodes.set(path, { kind: 'directory', bytes: bytes(''), mode, ...this.stamp() })
  }

  file(path: string, data: string | readonly number[], mode = 0o644): void {
    this.nodes.set(path, { kind: 'file', bytes: bytes(data), mode, ...this.stamp() })
  }

  other(path: string): void {
    this.nodes.set(path, { kind: 'other', bytes: bytes(''), mode: 0o600, ...this.stamp() })
  }

  symlink(path: string, target: string): void {
    this.nodes.set(path, { kind: 'symlink', bytes: bytes(target), mode: 0o777, ...this.stamp(), symlinkTarget: target })
  }

  mutate(path: string, data: string): void {
    const node = this.nodes.get(path) as RemoteNode
    node.bytes = bytes(data)
    Object.assign(node, this.stamp())
  }

  private stamp(): { inode: string; mtime: string } {
    const tick = ++this.clock
    return { inode: String(4096 + tick), mtime: `2026-08-13 12:00:00.${String(tick).padStart(9, '0')} +0000` }
  }

  private follow(path: string): RemoteNode | undefined {
    const node = this.nodes.get(path)
    if (node?.symlinkTarget === undefined) return node
    return this.nodes.get(node.symlinkTarget)
  }

  private statFields(node: RemoteNode): string {
    const word = node.kind === 'file' && node.bytes.byteLength === 0 ? 'regular empty file' : FILE_TYPE_WORDS[node.kind]
    const raw = (TYPE_BITS[node.kind] | node.mode).toString(16)
    return nulFields(word, String(node.bytes.byteLength), raw, node.inode, node.mtime)
  }

  private statCommand(args: readonly string[]): CommandResult {
    const follow = args[0] === '-L'
    expect(args.slice(follow ? 1 : 0, -1)).toEqual(['--printf', STAT_FORMAT, '--'])
    const path = args.at(-1) as string
    if (this.stagedProbe !== undefined && posix.basename(posix.dirname(path)).startsWith('.dsh-')) {
      const staged = this.stagedProbe
      this.stagedProbe = undefined
      if (staged === 'missing') return notFound('stat', path)
      return ok(this.statFields({ kind: 'directory', bytes: bytes(''), mode: 0o700, inode: '9', mtime: 'staged' }))
    }
    if (this.statOutput !== undefined) {
      const stdout = this.statOutput
      this.statOutput = undefined
      return ok(stdout)
    }
    const node = follow ? this.follow(path) : this.nodes.get(path)
    if (node === undefined) return notFound('stat', path)
    return ok(this.statFields(node))
  }

  private canonicalCommand(path: string): CommandResult {
    if (this.canonicalOutput !== undefined) {
      const stdout = this.canonicalOutput
      this.canonicalOutput = undefined
      return ok(stdout)
    }
    return ok(encode(`${this.nodes.get(path)?.symlinkTarget ?? path}\0`))
  }

  private listingCommand(path: string): CommandResult {
    if (this.listingOutput !== undefined) {
      const stdout = this.listingOutput
      this.listingOutput = undefined
      return ok(stdout)
    }
    let listed = ''
    for (const [child, node] of this.nodes) {
      if (child === path || posix.dirname(child) !== path) continue
      const followed = this.follow(child)
      listed += nulFields(child)
      listed += this.statFields(node)
      listed += followed === undefined ? DANGLING_FOLLOW : this.statFields(followed)
      listed += nulFields(node.symlinkTarget ?? child)
    }
    return ok(encode(listed))
  }

  private publishCommand(temporary: string, target: string): CommandResult {
    if (this.publicationOutput !== undefined) {
      const result = this.publicationOutput
      this.publicationOutput = undefined
      return result
    }
    if (this.competitor?.path === target) {
      const competitor = this.competitor
      this.competitor = undefined
      if (competitor.kind === 'directory') this.dir(competitor.path)
      else this.file(competitor.path, competitor.text)
    }
    if (this.nodes.has(target)) return ok('exists')
    const node = this.nodes.get(temporary)
    if (node === undefined) return notFound('ln', temporary)
    this.nodes.set(target, node)
    this.abortOnPublish?.abort('after commit')
    return ok('created')
  }

  private bashCommand(args: readonly string[]): CommandResult {
    expect(args[0]).toBe('-c')
    expect(args[2]).toBe('dsh-fs-tensorlake')
    const script = args[1] as string
    if (script.includes('realpath -mz -- "$1" | base64')) return this.canonicalCommand(args[3] as string)
    if (script.includes('find "$1"')) return this.listingCommand(args[3] as string)
    if (script.includes('ln -T')) return this.publishCommand(args[3] as string, args[4] as string)
    throw new Error(`unexpected control script: ${script}`)
  }

  private mkdirCommand(args: readonly string[]): CommandResult {
    expect(args[0]).toBe('--')
    const path = args[1] as string
    if (this.nodes.has(path)) {
      return { exitCode: 1, stdout: '', stderr: `mkdir: cannot create directory '${path}': File exists` }
    }
    this.dir(path, 0o755)
    return ok()
  }

  private chmodCommand(args: readonly string[]): CommandResult {
    expect(args[1]).toBe('--')
    const path = args[2] as string
    const node = this.nodes.get(path)
    if (node === undefined) return notFound('chmod', path)
    node.mode = Number.parseInt(args[0] as string, 8)
    return ok()
  }

  private moveCommand(args: readonly string[]): CommandResult {
    expect(args.slice(0, 2)).toEqual(['-T', '--'])
    const from = args[2] as string
    const to = args[3] as string
    const node = this.nodes.get(from)
    if (node === undefined) return notFound('mv', from)
    this.nodes.delete(from)
    this.nodes.set(to, node)
    this.abortOnPublish?.abort('after commit')
    return ok()
  }

  private removeCommand(args: readonly string[]): CommandResult {
    expect(args.slice(0, 2)).toEqual(['-rf', '--'])
    const path = args[2] as string
    for (const candidate of this.nodes.keys()) {
      if (candidate === path || candidate.startsWith(`${path}/`)) this.nodes.delete(candidate)
    }
    return ok()
  }

  readonly sandbox = {
    run: async (command: string, options?: RunOptions): Promise<CommandResult> => {
      const args = options?.args ?? []
      this.commands.push({ command, args })
      const failure = this.failures.get(command)
      if (failure !== undefined) {
        this.failures.delete(command)
        if (failure instanceof Error) throw failure
        return failure
      }
      switch (command) {
        case 'stat': return this.statCommand(args)
        case 'bash': return this.bashCommand(args)
        case 'mkdir': return this.mkdirCommand(args)
        case 'chmod': return this.chmodCommand(args)
        case 'mv': return this.moveCommand(args)
        case 'rm': return this.removeCommand(args)
        default: throw new Error(`unexpected control command: ${command}`)
      }
    },
    readFile: async (path: string): Promise<Uint8Array> => {
      this.reads.push(path)
      this.abortOnRead?.abort('mid-read')
      if (this.nextReadError !== undefined) {
        const error = this.nextReadError
        this.nextReadError = undefined
        throw error
      }
      if (this.grownRead !== undefined) {
        const grown = this.grownRead
        this.grownRead = undefined
        return grown
      }
      const node = this.follow(path)
      if (node === undefined) throw new RemoteAPIError(404, `no such file: ${path}`)
      return node.bytes.slice()
    },
    writeFile: async (path: string, content: Uint8Array): Promise<void> => {
      if (this.nextWriteError !== undefined) {
        const error = this.nextWriteError
        this.nextWriteError = undefined
        throw error
      }
      const parent = posix.dirname(path)
      if (!this.nodes.has(parent)) throw new RemoteAPIError(404, `no such directory: ${parent}`)
      this.nodes.set(path, { kind: 'file', bytes: content.slice(), mode: 0o644, ...this.stamp() })
      this.writes.push({ path, text: new TextDecoder().decode(content) })
    },
  } as unknown as Sandbox
}

async function setup(remote = new FakeSandbox()): Promise<{ ctx: Context; fs: TensorlakeFileSystem; remote: FakeSandbox }> {
  const ctx = new Context()
  const runtime = {
    cwd: '/workspace',
    runtimeRoot: '/workspace/.dsh-tensorlake',
    getSandbox: async () => remote.sandbox,
  } as unknown as TensorlakeRuntime
  ctx.provide('tensorlake', runtime)
  await ctx.plugin(TensorlakeFileSystem)
  return { ctx, fs: ctx.fs as TensorlakeFileSystem, remote }
}

async function expectCode(promise: Promise<unknown>, code: string): Promise<void> {
  await expect(promise).rejects.toMatchObject({ code })
}

async function drain(stream: AsyncIterable<string>): Promise<string> {
  let text = ''
  for await (const chunk of stream) text += chunk
  return text
}

/** Text of one committed remote file. */
function contentOf(remote: FakeSandbox, path: string): string {
  return new TextDecoder().decode(remote.nodes.get(path)?.bytes)
}

/** Commands of one argv name, in the order the provider issued them. */
function issued(remote: FakeSandbox, command: string): Array<readonly string[]> {
  return remote.commands.filter(entry => entry.command === command).map(entry => entry.args)
}

describe('TensorlakeFileSystem identity, metadata, and reads', () => {
  it('resolves remote paths, reports entry kinds, and lists direct children in stable order', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/z.txt', 'z')
    remote.file('/workspace/a.txt', 'a')
    remote.dir('/workspace/dir')
    remote.dir('/workspace/hollow')
    remote.other('/workspace/special')
    remote.file('/workspace/empty.txt', '')
    remote.file('/workspace/dir/nested.txt', 'nested')
    remote.symlink('/workspace/link.txt', '/workspace/a.txt')
    remote.symlink('/workspace/dangling.txt', '/workspace/gone.txt')
    const { fs } = await setup(remote)

    const link = await fs.resolve('link.txt')
    expect(link).toEqual({ targetKey: '/workspace/a.txt', displayPath: '/workspace/link.txt' })
    await expect(fs.lstat('link.txt')).resolves.toEqual(expect.objectContaining({ type: 'symlink' }))
    await expect(fs.lstat('a.txt')).resolves.toMatchObject({ type: 'file', size: 1 })
    await expect(fs.lstat('empty.txt')).resolves.toMatchObject({ type: 'file', size: 0 })
    await expect(fs.lstat('dir')).resolves.toEqual(expect.objectContaining({ type: 'directory' }))
    await expect(fs.lstat('special')).resolves.toEqual(expect.objectContaining({ type: 'other' }))
    await expect(fs.lstat('missing')).resolves.toBeUndefined()
    await expect(fs.stat(link)).resolves.toMatchObject({ type: 'file', size: 1 })
    await expect(fs.stat(await fs.resolve('dangling.txt'))).resolves.toBeUndefined()

    const listed = await fs.listDir(await fs.resolve('.'))
    expect(listed.map(entry => entry.name))
      .toEqual(['a.txt', 'dangling.txt', 'dir', 'empty.txt', 'hollow', 'link.txt', 'special', 'z.txt'])
    await expect(fs.listDir(await fs.resolve('hollow'))).resolves.toEqual([])
    expect(listed.find(entry => entry.name === 'dir')).toEqual(expect.objectContaining({ type: 'directory' }))
    expect(listed.find(entry => entry.name === 'dir')?.size).toBeUndefined()
    expect(listed.find(entry => entry.name === 'link.txt')).toMatchObject({
      type: 'file',
      size: 1,
      target: { targetKey: '/workspace/a.txt', displayPath: '/workspace/link.txt' },
    })
    expect(listed.find(entry => entry.name === 'special')?.version).toBeDefined()
    expect(listed.find(entry => entry.name === 'dangling.txt')).toEqual({
      name: 'dangling.txt',
      type: 'other',
      target: { targetKey: '/workspace/gone.txt', displayPath: '/workspace/dangling.txt' },
    })
    expect(listed.some(entry => entry.name === 'nested.txt')).toBe(false)
  })

  it('projects canonical process paths, file URLs, and containment', async () => {
    const remote = new FakeSandbox()
    remote.dir('/workspace/nested')
    remote.file('/workspace/nested/multibyte # file.ts', 'text')
    remote.file('/outside.ts', 'outside')
    const { fs } = await setup(remote)
    const root = await fs.resolve('/')
    const workspace = await fs.resolve('/workspace')
    const nested = await fs.resolve('/workspace/nested/multibyte # file.ts')
    const outside = await fs.resolve('/outside.ts')

    expect(fs.processPath(nested)).toBe('/workspace/nested/multibyte # file.ts')
    expect(fs.fileUrl(nested)).toBe('file:///workspace/nested/multibyte%20%23%20file.ts')
    expect(fs.contains(workspace, workspace)).toBe(true)
    expect(fs.contains(workspace, nested)).toBe(true)
    expect(fs.contains(nested, workspace)).toBe(false)
    expect(fs.contains(workspace, outside)).toBe(false)
    expect(fs.contains(root, outside)).toBe(true)
    expect(() => fs.fileUrl({ targetKey: FsTargetKey('relative'), displayPath: 'relative' }))
      .toThrow('expected an absolute process path')
  })

  it('resolves relative paths against an explicit cwd override', async () => {
    const remote = new FakeSandbox()
    remote.dir('/other')
    remote.file('/other/file.txt', 'other')
    const { fs } = await setup(remote)

    await expect(fs.resolve('file.txt', { cwd: '/other' }))
      .resolves.toEqual({ targetKey: '/other/file.txt', displayPath: '/other/file.txt' })
    await expect(fs.lstat('file.txt', { cwd: '/other' })).resolves.toMatchObject({ type: 'file', size: 5 })
  })

  it('preserves newline and multibyte canonical paths through strict ASCII framing', async () => {
    const remote = new FakeSandbox()
    const path = '/workspace/你好\nfile.ts'
    remote.file(path, 'text')
    const { fs } = await setup(remote)

    await expect(fs.resolve(path)).resolves.toEqual({ targetKey: path, displayPath: path })
  })

  it.each([
    ['invalid base64', '!!!!'],
    ['non-canonical base64', 'AB=='],
    ['empty output', ''],
    ['missing terminator', encode('/workspace/file')],
    ['a bare terminator', encode('\0')],
    ['multiple records', encode('/workspace/file\0/other\0')],
    ['invalid UTF-8', encode([47, 0xff, 0])],
    ['a relative path', encode('workspace/file\0')],
  ])('rejects %s from the canonical path transport', async (_label, output) => {
    const remote = new FakeSandbox()
    remote.canonicalOutput = output
    const { fs } = await setup(remote)

    await expectCode(fs.resolve('file'), 'FS_IO_ERROR')
  })

  it.each([
    ['invalid base64', '!!!!'],
    ['invalid UTF-8', encode([0xff, 0])],
    ['a missing terminator', encode('/workspace/a')],
    ['a partial record', encode(nulFields('/workspace/a'))],
    ['a non-absolute entry path', encode(listingRecord('a.txt', '/workspace/a.txt'))],
    ['a non-absolute canonical path', encode(listingRecord('/workspace/a.txt', 'a.txt'))],
    ['invalid numeric facts', encode(listingRecord('/workspace/a.txt', '/workspace/a.txt', 'not-a-size'))],
  ])('rejects %s from the listing transport', async (_label, output) => {
    const remote = new FakeSandbox()
    const { fs } = await setup(remote)
    remote.listingOutput = output

    await expectCode(fs.listDir(await fs.resolve('/workspace')), 'FS_IO_ERROR')
  })

  it.each([
    ['a missing terminator', nulFields('regular file', '1', '81a4', '12').slice(0, -1)],
    ['a malformed field group', nulFields('regular file', '1', '81a4')],
    ['a non-numeric size', nulFields('regular file', 'x', '81a4', '12', 'now')],
    ['an unrepresentable size', nulFields('regular file', '99999999999999999999', '81a4', '12', 'now')],
    ['a non-hexadecimal mode', nulFields('regular file', '1', 'zz', '12', 'now')],
    ['an unrepresentable mode', nulFields('regular file', '1', 'fffffffffffffffff', '12', 'now')],
    ['a non-numeric inode', nulFields('regular file', '1', '81a4', 'x', 'now')],
    ['an empty mtime', nulFields('regular file', '1', '81a4', '12', '')],
  ])('rejects %s from the stat transport', async (_label, output) => {
    const remote = new FakeSandbox()
    remote.file('/workspace/a.txt', 'a')
    const { fs } = await setup(remote)
    const target = await fs.resolve('a.txt')
    remote.statOutput = output

    await expectCode(fs.stat(target), 'FS_IO_ERROR')
  })

  it('reads whole UTF-8 files and streams one validated chunk', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/text.txt', 'A€B')
    remote.file('/workspace/empty.txt', '')
    const { fs } = await setup(remote)

    await expect(fs.readText(await fs.resolve('text.txt'))).resolves.toBe('A€B')
    expect(await drain(await fs.streamText(await fs.resolve('text.txt')))).toBe('A€B')
    expect(await drain(await fs.streamText(await fs.resolve('empty.txt')))).toBe('')
  })

  it('matches the seam binary sample while edits reject any NUL byte', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/late-nul.txt', `${'a'.repeat(8192)}\0tail`)
    remote.file('/workspace/binary', [0, 1])
    remote.file('/workspace/invalid', [0xff])
    const { fs } = await setup(remote)
    const late = await fs.resolve('late-nul.txt')

    await expect(fs.readText(late)).resolves.toContain('\0tail')
    await expectCode(fs.editText(late, { oldString: 'tail', newString: 'end', replaceAll: false }), 'FS_NOT_TEXT')
    await expectCode(fs.readText(await fs.resolve('binary')), 'FS_NOT_TEXT')
    await expectCode(fs.readText(await fs.resolve('invalid')), 'FS_NOT_TEXT')
  })

  it('maps missing, non-regular, and racing read failures', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/text.txt', 'text')
    remote.dir('/workspace/directory')
    const { fs } = await setup(remote)
    const target = await fs.resolve('text.txt')

    await expectCode(fs.readText(await fs.resolve('missing')), 'FS_NOT_FOUND')
    await expectCode(fs.readText(await fs.resolve('directory')), 'FS_NOT_REGULAR_FILE')
    await expectCode(fs.streamText(await fs.resolve('missing')), 'FS_NOT_FOUND')

    const raced = await fs.streamText(target)
    remote.nextReadError = new RemoteAPIError(404, 'gone after stat')
    await expectCode(drain(raced), 'FS_NOT_FOUND')
  })

  it('readBytes enforces the stat preflight, bounds a post-stat grower, and returns raw bytes', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/img.bin', [0x89, 0, 0xff, 0x47])
    remote.dir('/workspace/directory')
    const { fs } = await setup(remote)
    const target = await fs.resolve('img.bin')

    expect(Array.from(await fs.readBytes(target, undefined, 4))).toEqual([0x89, 0, 0xff, 0x47])
    expect(remote.reads).toEqual(['/workspace/img.bin'])
    remote.reads.length = 0
    await expectCode(fs.readBytes(target, undefined, 3), 'FS_TOO_LARGE')
    expect(remote.reads).toEqual([])

    remote.grownRead = bytes([1, 1, 1, 1, 1])
    await expectCode(fs.readBytes(target, undefined, 4), 'FS_TOO_LARGE')
    await expectCode(fs.readBytes(await fs.resolve('missing'), undefined, 4), 'FS_NOT_FOUND')
    await expectCode(fs.readBytes(await fs.resolve('directory'), undefined, 4), 'FS_NOT_REGULAR_FILE')
  })

  it('honors aborts before and between remote round-trips', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/a.txt', 'a')
    const { fs } = await setup(remote)
    const target = await fs.resolve('a.txt')

    await expectCode(fs.resolve('a.txt', { signal: AbortSignal.abort() }), 'FS_ABORTED')
    await expectCode(fs.lstat('a.txt', undefined, AbortSignal.abort()), 'FS_ABORTED')
    await expectCode(fs.stat(target, AbortSignal.abort()), 'FS_ABORTED')

    const observed = new AbortController()
    remote.abortOnRead = observed
    await expectCode(fs.readText(target, observed.signal), 'FS_ABORTED')

    const failing = new AbortController()
    remote.abortOnRead = failing
    remote.nextReadError = new Error('transport closed')
    await expectCode(fs.readText(target, failing.signal), 'FS_ABORTED')
  })

  it('rejects empty paths and directory-listing type errors', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'x')
    const { fs } = await setup(remote)

    await expectCode(fs.resolve('   '), 'FS_NOT_FOUND')
    await expectCode(fs.lstat(''), 'FS_NOT_FOUND')
    await expectCode(fs.listDir(await fs.resolve('missing')), 'FS_NOT_FOUND')
    await expectCode(fs.listDir(await fs.resolve('file.txt')), 'FS_NOT_DIRECTORY')
    const workspace = await fs.resolve('/workspace')
    remote.failures.set('bash', new Error('listing transport failed'))
    await expectCode(fs.listDir(workspace), 'FS_IO_ERROR')
  })

  it('maps proxy, canonicalization, permission, and generic control failures', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/a.txt', 'a')
    const { fs } = await setup(remote)
    const target = await fs.resolve('a.txt')

    remote.failures.set('bash', { exitCode: 1, stdout: '', stderr: "realpath: '/x': Permission denied" })
    await expectCode(fs.resolve('denied'), 'FS_PERMISSION_DENIED')
    remote.failures.set('bash', { exitCode: 1, stdout: '', stderr: '' })
    await expectCode(fs.resolve('silent'), 'FS_IO_ERROR')
    remote.failures.set('bash', new Error('canonical transport failed'))
    await expectCode(fs.resolve('broken'), 'FS_IO_ERROR')

    remote.failures.set('stat', { exitCode: 1, stdout: '', stderr: 'stat: Input/output error' })
    await expectCode(fs.stat(target), 'FS_IO_ERROR')
    remote.failures.set('stat', { exitCode: 1, stdout: '', stderr: "stat: cannot read '/workspace/a.txt': Permission denied" })
    await expectCode(fs.stat(target), 'FS_PERMISSION_DENIED')

    remote.nextReadError = new RemoteAPIError(404, 'vanished')
    await expectCode(fs.readText(target), 'FS_NOT_FOUND')
    remote.nextReadError = 'transport vanished'
    await expectCode(fs.readText(target), 'FS_IO_ERROR')
  })
})

describe('TensorlakeFileSystem atomic writes and edits', () => {
  it('creates owner-only files through guarded publication and returns the committed version', async () => {
    const { fs, remote } = await setup()
    const target = await fs.resolve('new.txt')

    const outcome = await fs.writeText(target, 'one\r\ntwo\rthree', { kind: 'createIfAbsent' })

    expect(outcome).toMatchObject({ operation: 'create', before: null, after: 'one\ntwo\rthree' })
    expect(contentOf(remote, '/workspace/new.txt')).toBe('one\r\ntwo\rthree')
    expect(remote.nodes.get('/workspace/new.txt')?.mode).toBe(0o600)
    const staging = posix.dirname(remote.writes[0]!.path)
    expect(posix.dirname(staging)).toBe('/workspace')
    expect(issued(remote, 'mkdir')).toEqual([['--', staging]])
    expect(issued(remote, 'chmod')).toEqual([['700', '--', staging], ['600', '--', `${staging}/content`]])
    expect(issued(remote, 'rm')).toEqual([['-rf', '--', staging]])
    expect(issued(remote, 'mv')).toEqual([])
    expect(remote.nodes.has(staging)).toBe(false)
    await expect(fs.stat(target)).resolves.toMatchObject({ version: outcome.version, size: 14 })
  })

  it('preserves replacement mode, normalizes only CRLF for diffs, and changes version on external writes', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'old\r\nline\rlone', 0o640)
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')
    const version = (await fs.stat(target))!.version

    const outcome = await fs.writeText(target, 'new', { kind: 'replaceIfVersion', version })

    expect(outcome).toMatchObject({ operation: 'update', before: 'old\nline\rlone', after: 'new' })
    expect(remote.nodes.get('/workspace/file.txt')?.mode).toBe(0o640)
    expect(issued(remote, 'mv')).toHaveLength(1)
    await expect(fs.stat(target)).resolves.toMatchObject({ version: outcome.version })
    remote.mutate('/workspace/file.txt', 'external')
    expect((await fs.stat(target))!.version).not.toBe(outcome.version)
  })

  it('returns null as the overwrite diff basis for binary prior content', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', [0xff, 0])
    const { fs } = await setup(remote)

    await expect(fs.writeText(await fs.resolve('file.txt'), 'valid'))
      .resolves.toMatchObject({ operation: 'update', before: null, after: 'valid' })
  })

  it('fails an overwrite when reading its diff basis fails for another reason', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'prior')
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')
    remote.nextReadError = new Error('read transport failed')

    await expectCode(fs.writeText(target, 'replacement'), 'FS_IO_ERROR')
    expect(contentOf(remote, '/workspace/file.txt')).toBe('prior')
  })

  it('enforces create and version intents before publication', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'v1')
    remote.dir('/workspace/dir')
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')
    const version = (await fs.stat(target))!.version

    await expectCode(fs.writeText(target, 'blind', { kind: 'createIfAbsent' }), 'FS_NOT_OBSERVED')
    remote.mutate('/workspace/file.txt', 'v2')
    await expectCode(fs.writeText(target, 'stale', { kind: 'replaceIfVersion', version }), 'FS_STALE_VERSION')
    await expectCode(
      fs.writeText(await fs.resolve('missing'), 'stale', { kind: 'replaceIfVersion', version }),
      'FS_STALE_VERSION',
    )
    await expectCode(fs.writeText(await fs.resolve('dir'), 'x'), 'FS_NOT_REGULAR_FILE')
    expect(issued(remote, 'mkdir')).toEqual([])
  })

  it.each([
    ['file', 'competitor'],
    ['directory', ''],
  ] as const)('preserves a %s created after the guarded-create probe', async (kind, text) => {
    const remote = new FakeSandbox()
    remote.competitor = { path: '/workspace/race', kind, text }
    const { fs } = await setup(remote)

    await expectCode(fs.writeText(await fs.resolve('race'), 'ours', { kind: 'createIfAbsent' }), 'FS_NOT_OBSERVED')
    expect(remote.nodes.get('/workspace/race')?.kind).toBe(kind)
    expect(contentOf(remote, '/workspace/race')).toBe(text)
    expect(issued(remote, 'rm')).toHaveLength(1)
  })

  it.each([
    ['an invalid', { exitCode: 0, stdout: 'unexpected', stderr: '' }],
    ['a failed', { exitCode: 1, stdout: '', stderr: 'ln: Input/output error' }],
  ])('rejects %s guarded publication before claiming success', async (_label, publication) => {
    const remote = new FakeSandbox()
    remote.publicationOutput = publication
    const { fs } = await setup(remote)

    await expectCode(fs.writeText(await fs.resolve('invalid.txt'), 'ours', { kind: 'createIfAbsent' }), 'FS_IO_ERROR')
    expect(remote.nodes.has('/workspace/invalid.txt')).toBe(false)
    expect(issued(remote, 'rm')).toHaveLength(1)
  })

  it.each(['missing', 'directory'] as const)('rejects staged content that is %s before publication', async (staged) => {
    const remote = new FakeSandbox()
    const { fs } = await setup(remote)
    remote.stagedProbe = staged

    await expectCode(fs.writeText(await fs.resolve('staged.txt'), 'ours'), 'FS_IO_ERROR')
    expect(remote.nodes.has('/workspace/staged.txt')).toBe(false)
    expect(issued(remote, 'rm')).toHaveLength(1)
  })

  it.each([
    ['a replacement move', undefined],
    ['a guarded create', { kind: 'createIfAbsent' } as const],
  ])('does not turn an abort observed after %s into a failed write', async (_label, intent) => {
    const remote = new FakeSandbox()
    const controller = new AbortController()
    remote.abortOnPublish = controller
    const { fs } = await setup(remote)

    await expect(fs.writeText(await fs.resolve('committed.txt'), 'yes', intent, controller.signal))
      .resolves.toMatchObject({ operation: 'create' })
    expect(controller.signal.aborted).toBe(true)
    expect(contentOf(remote, '/workspace/committed.txt')).toBe('yes')
  })

  it('does not turn post-commit staging cleanup failure into a failed write', async () => {
    const remote = new FakeSandbox()
    const { fs } = await setup(remote)
    remote.failures.set('rm', new Error('committed staging cleanup failed'))

    await expect(fs.writeText(await fs.resolve('committed.txt'), 'yes')).resolves.toMatchObject({ operation: 'create' })
    expect(contentOf(remote, '/workspace/committed.txt')).toBe('yes')
  })

  it('cleans staging and maps upload, permission, not-found, and cleanup-failing publications', async () => {
    const remote = new FakeSandbox()
    const { fs } = await setup(remote)

    remote.nextWriteError = new Error('upload transport failed')
    await expectCode(fs.writeText(await fs.resolve('upload.txt'), 'x'), 'FS_IO_ERROR')
    expect(issued(remote, 'rm')).toHaveLength(1)

    remote.failures.set('chmod', { exitCode: 1, stdout: '', stderr: 'chmod: changing permissions: Operation not permitted' })
    await expectCode(fs.writeText(await fs.resolve('mode.txt'), 'x'), 'FS_PERMISSION_DENIED')

    remote.failures.set('mv', { exitCode: 1, stdout: '', stderr: "mv: cannot stat '/tmp/x': No such file or directory" })
    await expectCode(fs.writeText(await fs.resolve('gone.txt'), 'x'), 'FS_NOT_FOUND')

    remote.failures.set('mv', { exitCode: 1, stdout: '', stderr: 'mv: Input/output error' })
    remote.failures.set('rm', new Error('staging cleanup also failed'))
    await expectCode(fs.writeText(await fs.resolve('broken.txt'), 'x'), 'FS_IO_ERROR')
    expect(remote.nodes.has('/workspace/broken.txt')).toBe(false)
  })

  it('leaves no cleanup behind when the private staging directory cannot be created', async () => {
    const remote = new FakeSandbox()
    const { fs } = await setup(remote)
    remote.failures.set('mkdir', { exitCode: 1, stdout: '', stderr: "mkdir: cannot create directory '/workspace/.dsh': File exists" })

    await expectCode(fs.writeText(await fs.resolve('collision.txt'), 'x'), 'FS_IO_ERROR')
    expect(issued(remote, 'rm')).toEqual([])
  })

  it('applies literal edits atomically and restores the detected CRLF style', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'one\r\ntwo\r\nthree\n')
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')
    const version = (await fs.stat(target))!.version

    const outcome = await fs.editText(target, { oldString: 'two\r\n', newString: 'TWO\r\n', replaceAll: false }, { version })

    expect(outcome).toMatchObject({ before: 'one\ntwo\nthree\n', after: 'one\nTWO\nthree\n' })
    expect(contentOf(remote, '/workspace/file.txt')).toBe('one\r\nTWO\r\nthree\r\n')
    await expect(fs.stat(target)).resolves.toMatchObject({ version: outcome.version })
  })

  it('reports stale and literal-match failures with stable codes', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'a a')
    remote.other('/workspace/special')
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')

    await expectCode(fs.editText(target, { oldString: '', newString: 'x', replaceAll: false }), 'FS_EDIT_NOT_FOUND')
    await expectCode(fs.editText(target, { oldString: 'z', newString: 'x', replaceAll: false }), 'FS_EDIT_NOT_FOUND')
    await expectCode(fs.editText(target, { oldString: 'a', newString: 'x', replaceAll: false }), 'FS_AMBIGUOUS_EDIT')
    await expect(fs.editText(target, { oldString: 'a', newString: 'x', replaceAll: true }))
      .resolves.toMatchObject({ before: 'a a', after: 'x x' })
    expect(contentOf(remote, '/workspace/file.txt')).toBe('x x')
    await expectCode(
      fs.editText(target, { oldString: 'x', newString: 'y', replaceAll: false }, { version: FsVersion('stale') }),
      'FS_STALE_VERSION',
    )
    await expectCode(
      fs.editText(await fs.resolve('missing'), { oldString: 'x', newString: 'y', replaceAll: false }),
      'FS_STALE_VERSION',
    )
    await expectCode(
      fs.editText(await fs.resolve('special'), { oldString: 'x', newString: 'y', replaceAll: false }),
      'FS_NOT_REGULAR_FILE',
    )
  })

  it('serializes guarded mutations per target so only one stale version can win', async () => {
    const remote = new FakeSandbox()
    remote.file('/workspace/file.txt', 'base')
    const { fs } = await setup(remote)
    const target = await fs.resolve('file.txt')
    const version = (await fs.stat(target))!.version

    const results = await Promise.allSettled([
      fs.writeText(target, 'one', { kind: 'replaceIfVersion', version }),
      fs.editText(target, { oldString: 'base', newString: 'two', replaceAll: false }, { version }),
    ])

    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1)
    expect(results.filter(result => result.status === 'rejected')).toHaveLength(1)
  })
})
