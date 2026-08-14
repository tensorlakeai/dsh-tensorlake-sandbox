/**
 * Byte-faithful output recovery for the Tensorlake subprocess adapter. The
 * sandbox daemon frames captured output as newline-stripped text lines and
 * flushes partial lines indistinguishably, so remote encoders base64 each
 * stream and end it with a reserved marker; concatenating the delivered line
 * texts in order reproduces the encoder's exact character stream.
 */

import { Buffer } from 'node:buffer'
import type { SubprocessOutputRead, SubprocessOutputReader } from '@deepseek-ai/dsh-subprocess'

/** Reserved non-base64 marker proving that one remote encoder reached clean EOF. */
export const OUTPUT_COMPLETE_MARKER = '!dsh-tensorlake-output-complete!'

/**
 * Remote per-stream encoder run under `node -e`. Coreutils filters block-buffer
 * pipe output, which would hold small writes hostage to a descendant keeping
 * the stream open; this encoder flushes every available chunk immediately.
 * Each emitted line is a whole 3-byte quantum, so the concatenated line texts
 * form one continuous base64 stream with padding only at EOF, and the marker
 * is written only after clean end-of-input.
 */
export const OUTPUT_ENCODER_SOURCE = [
  '(async () => {',
  '  const write = async (text) => {',
  "    if (!process.stdout.write(text)) await new Promise(resolve => process.stdout.once('drain', resolve))",
  '  }',
  '  let carry = Buffer.alloc(0)',
  '  for await (const chunk of process.stdin) {',
  '    const data = carry.length === 0 ? chunk : Buffer.concat([carry, chunk])',
  '    const usable = data.length - (data.length % 3)',
  '    carry = data.subarray(usable)',
  "    if (usable > 0) await write(data.subarray(0, usable).toString('base64') + '\\n')",
  '  }',
  "  if (carry.length > 0) await write(carry.toString('base64') + '\\n')",
  `  await write(${JSON.stringify(OUTPUT_COMPLETE_MARKER)} + '\\n')`,
  '})().catch(() => { process.exitCode = 1 })',
].join('\n')

const BASE64_CHARS = /^[A-Za-z0-9+/=]*$/

/**
 * Incrementally decode one remote encoder's base64 character stream. Text
 * arrives in arbitrary splits, so decoding consumes complete 4-character
 * quanta and retains the remainder until the next push or the marker.
 */
export class TensorlakeOutputDecoder {
  private pending = ''
  private complete = false

  /** Whether this stream's completion marker has arrived. */
  get isComplete(): boolean {
    return this.complete
  }

  /**
   * Consume one delivered line text and decode every available quantum.
   * @param text - Newline-stripped transport text in delivery order.
   * @returns the raw bytes this push made available.
   */
  push(text: string): Buffer {
    if (text.length === 0) return Buffer.alloc(0)
    if (this.complete) throw new Error('subprocess-tensorlake: output transport continued after completion')
    this.pending += text
    const markerStart = this.pending.indexOf('!')
    let decodable: string
    if (markerStart < 0) {
      const whole = this.pending.length - (this.pending.length % 4)
      decodable = this.pending.slice(0, whole)
      this.pending = this.pending.slice(whole)
    } else {
      decodable = this.pending.slice(0, markerStart)
      const tail = this.pending.slice(markerStart)
      if (tail === OUTPUT_COMPLETE_MARKER) {
        this.complete = true
        this.pending = ''
      } else if (OUTPUT_COMPLETE_MARKER.startsWith(tail)) {
        this.pending = tail
      } else {
        throw new Error('subprocess-tensorlake: invalid output transport marker')
      }
    }
    if (decodable.length === 0) return Buffer.alloc(0)
    if (decodable.length % 4 !== 0 || !BASE64_CHARS.test(decodable)) {
      throw new Error('subprocess-tensorlake: invalid base64 output transport')
    }
    const bytes = Buffer.from(decodable, 'base64')
    if (bytes.toString('base64').replaceAll('=', '') !== decodable.replaceAll('=', '')) {
      throw new Error('subprocess-tensorlake: invalid base64 output transport')
    }
    return bytes
  }

  /**
   * Validate clean encoder completion, or discard an interrupted trailing
   * remainder after requested termination or drain expiry.
   * @param requireComplete - Whether natural completion requires the reserved marker.
   */
  finish(requireComplete = true): void {
    if (!requireComplete) {
      this.pending = ''
      return
    }
    if (this.pending.length > 0) {
      throw new Error('subprocess-tensorlake: truncated base64 output transport')
    }
    if (!this.complete) throw new Error('subprocess-tensorlake: incomplete output transport')
  }
}

/** Offset reader for one collect-mode Tensorlake stream. */
export class TensorlakeOutputReader implements SubprocessOutputReader {
  private retained = Buffer.alloc(0)
  private totalBytes = 0
  private spillIntact = true

  /**
   * Create a bounded reader over one remote spill path.
   * @param maxBytes - In-memory tail cap.
   * @param maxSpillBytes - Maximum complete remote file size the caller accepts.
   * @param spillPath - Remote full-output path.
   */
  constructor(
    private readonly maxBytes: number,
    private readonly maxSpillBytes: number | undefined,
    private readonly spillPath: string,
  ) {}

  /** Total bytes observed from the transport. */
  get size(): number {
    return this.totalBytes
  }

  /** Stop advertising a remote spill whose writer did not reach clean EOF. */
  invalidateSpill(): void {
    this.spillIntact = false
  }

  /**
   * Append decoded raw bytes, keeping only the bounded tail in memory.
   * @param bytes - Raw command bytes recovered from the ASCII transport.
   */
  push(bytes: Uint8Array): void {
    if (bytes.length === 0) return
    this.totalBytes += bytes.length
    const combined = Buffer.concat([this.retained, bytes])
    this.retained = combined.length > this.maxBytes
      ? Buffer.from(combined.subarray(combined.length - this.maxBytes))
      : combined
  }

  /** @inheritdoc */
  readFrom(fromByte: number): SubprocessOutputRead {
    const firstRetained = this.totalBytes - this.retained.length
    const lossy = fromByte < firstRetained
    const start = lossy ? 0 : Math.min(this.retained.length, Math.max(0, fromByte - firstRetained))
    return {
      text: this.retained.subarray(start).toString('utf8'),
      nextOffset: this.totalBytes,
      lossy,
      ...(lossy && this.spillIntact && this.maxSpillBytes !== undefined && this.totalBytes <= this.maxSpillBytes
        ? { spillPath: this.spillPath }
        : {}),
    }
  }
}
