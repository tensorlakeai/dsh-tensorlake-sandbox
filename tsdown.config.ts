import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: {
    runtime: 'src/runtime.ts',
    'filesystem/index': 'src/filesystem/index.ts',
    'subprocess/index': 'src/subprocess/index.ts',
  },
  outDir: 'lib',
  format: ['esm'],
  fixedExtension: false,
  platform: 'node',
  target: 'es2024',
  dts: true,
  clean: true,
  sourcemap: true,
})
