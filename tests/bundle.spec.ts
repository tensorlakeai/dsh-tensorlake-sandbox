import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { applyEntryPatches, entryListSchema, type PatchOptions } from '@deepseek-ai/cordis-plugin-include'
import yaml from 'js-yaml'
import { describe, expect, it } from 'vitest'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

describe('installable bundle', () => {
  it('declares the shipped patch and all three service entry points', async () => {
    const manifest = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8')) as {
      dsh?: { bundle?: { patch?: string } }
      exports?: Record<string, unknown>
      files?: string[]
    }

    expect(manifest.dsh?.bundle?.patch).toBe('./cordis.patch.yml')
    expect(manifest.exports).toHaveProperty('./runtime')
    expect(manifest.exports).toHaveProperty('./filesystem')
    expect(manifest.exports).toHaveProperty('./subprocess')
    expect(manifest.files).toContain('lib/')
    expect(manifest.files).toContain('cordis.patch.yml')
  })

  it('disables the local providers and inserts one shared Tensorlake world', async () => {
    const patches = yaml.load(await readFile(resolve(root, 'cordis.patch.yml'), 'utf8'), {
      schema: entryListSchema,
    }) as PatchOptions[]
    const warnings: string[] = []
    const entries = applyEntryPatches([
      { id: 'subprocess', name: '@deepseek-ai/dsh-subprocess-local' },
      { id: 'fs-sandbox', name: '@deepseek-ai/dsh-fs-sandbox' },
      {
        id: 'sandbox-policy',
        name: '@deepseek-ai/dsh-sandbox-policy',
        config: { mode: 'workspace-write', workspaceRoot: '/host' },
      },
      { id: 'bash-sandbox', name: '@deepseek-ai/dsh-bash-sandbox' },
      { id: 'approval', name: '@deepseek-ai/dsh-user-approval', config: { policy: 'ask' } },
      { id: 'permission', name: '@deepseek-ai/dsh-permission-presets' },
    ], patches, (message, ...args) => { warnings.push([message, ...args].join(' ')) })

    expect(warnings).toEqual([])
    expect(entries.find(entry => entry.id === 'subprocess')?.disabled).toBe(true)
    expect(entries.find(entry => entry.id === 'fs-sandbox')?.disabled).toBe(true)
    expect(entries.find(entry => entry.id === 'bash-sandbox')?.disabled).toBe(false)
    expect(entries.find(entry => entry.id === 'sandbox-policy')?.config).toEqual({
      mode: 'danger-full-access',
      workspaceRoot: { __jsExpr: "process.env.DSH_TENSORLAKE_CWD || '/home/tl-user/workspace'" },
    })
    expect(entries.find(entry => entry.id === 'tensorlake-runtime')?.config).toEqual({
      cwd: { __jsExpr: "process.env.DSH_TENSORLAKE_CWD || '/home/tl-user/workspace'" },
    })
    expect(entries.find(entry => entry.id === 'approval')?.config).toEqual({ policy: 'never' })
    expect(entries.find(entry => entry.id === 'permission')?.config).toEqual({
      presets: {
        'danger-full-access': { sandbox: 'danger-full-access', approval: 'never' },
      },
      defaultPreset: 'danger-full-access',
    })
    expect(entries.filter(entry => entry.id?.startsWith('tensorlake-')).map(entry => entry.name)).toEqual([
      '@tensorlake/dsh-sandbox/runtime',
      '@tensorlake/dsh-sandbox/subprocess',
      '@tensorlake/dsh-sandbox/filesystem',
    ])
  })
})
