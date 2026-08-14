# Tensorlake sandbox for DeepSeek Harness

`@tensorlake/dsh-sandbox` moves DeepSeek Harness file, subprocess, Bash, terminal, and LSP operations into one short-lived Tensorlake microVM. It is an installable dsh bundle and does not require changes to the Harness installation.

## Install

Install dsh and add this bundle to the profile you run:

```sh
npm install --global @deepseek-ai/dsh
dsh plugin --profile headless add @tensorlake/dsh-sandbox
TENSORLAKE_API_KEY=... DEEPSEEK_API_KEY=... dsh --profile headless "build and test this repo"
```

During development, install a local checkout from its directory:

```sh
dsh plugin --profile headless add .
```

Use `dsh --profile headless --dump-config` to verify that the `@tensorlake/dsh-sandbox` layer disables `subprocess`, `fs-sandbox`, and `bash-sandbox`, then inserts the Tensorlake runtime, subprocess, filesystem, and unconfined Bash rows.

## Configuration

The bundle starts an ephemeral sandbox on profile boot and terminates it when dsh exits. The runtime module accepts these Cordis config fields:

| Field | Default | Meaning |
|---|---:|---|
| `apiKey` | `TENSORLAKE_API_KEY` | Tensorlake API credential used only by the host SDK |
| `cwd` | `/workspace` | Absolute Linux working directory shared by file and process providers |
| `timeoutSecs` | `600` | Sandbox inactivity timeout |
| `cpus` | Tensorlake default | Virtual CPU allocation |
| `memoryMb` | Tensorlake default | Memory allocation in MiB |
| `diskMb` | Tensorlake default | Root disk allocation in MiB |

Override the inserted row in the profile's `cordis.patch.yml`; a patch replaces the complete config, so restate every non-default field you need:

```yaml
- id: tensorlake-runtime
  config:
    cwd: /workspace
    timeoutSecs: 1800
    cpus: 2
    memoryMb: 4096
```

`apiKey` is optional and should normally remain omitted. The package never copies `TENSORLAKE_API_KEY`, `DEEPSEEK_API_KEY`, other credential-shaped environment variables, or `DSH_*` variables into sandbox processes. A caller may still pass an explicit environment entry through a Harness tool or service request.

## Runtime requirements

The Tensorlake image must provide `bash`, Node.js, and GNU `base64`, `chmod`, `env`, `find`, `ln`, `mkdir`, `mv`, `ps`, `realpath`, `rm`, `stat`, and `tee`. The default managed Ubuntu image provides these tools.

The package targets `@deepseek-ai/dsh` `0.1.0-rc.6` or later compatible release. The dsh installation supplies its optional Cordis, filesystem, subprocess, and Schemastery peers through the profile module fallback. The package uses only public `ctx.fs` and `ctx.subprocess` service definitions; no DeepSeek Harness source registration, generated catalogs, or in-repository configuration is required.

## Known limitations

- `tensorlake@0.5.103`, the current SDK release, pins `undici@8.3.0` and `nanoid@3.3.11`; `npm audit --omit=dev` reports high-severity advisories for those transitive versions. No audit-clean current Tensorlake SDK release is available, so review the upstream advisories before production use and update the SDK pin when Tensorlake publishes one.

## Develop

```sh
npm install
npm run check
npm pack
```

The three Loader entry points are `@tensorlake/dsh-sandbox/runtime`, `@tensorlake/dsh-sandbox/filesystem`, and `@tensorlake/dsh-sandbox/subprocess`. Each module default-exports its service class; do not add function-plugin named exports to those modules because the Cordis Loader treats mixed export forms as a function-plugin namespace.
