# Tensorlake sandbox for DeepSeek Harness

`@tensorlakeai/dsh-sandbox` moves DeepSeek Harness file, subprocess, Bash, terminal, and LSP operations into one short-lived Tensorlake microVM. It is an installable dsh bundle and does not require changes to the Harness installation.

## Prerequisites

- Node.js `^22.19.0` or `>=24.0.0`
- `@deepseek-ai/dsh` `0.1.0-rc.6` or a later compatible release
- A Tensorlake project with `TENSORLAKE_API_KEY` set in the host environment
- `DEEPSEEK_API_KEY` set in the host environment for the default DeepSeek model provider

Keep credentials in environment variables or a secret manager; do not commit them to the profile or repository.

## Install

Install dsh and add this bundle to the profile you run:

```sh
npm install --global @deepseek-ai/dsh
dsh plugin --profile headless add @tensorlakeai/dsh-sandbox
TENSORLAKE_API_KEY=... DEEPSEEK_API_KEY=... dsh --profile headless "build and test this repo"
```

During development, install a local checkout from its directory:

```sh
npm install
npm run build
dsh plugin --profile headless add .
```

Use `dsh --profile headless --dump-config` to verify that the `@tensorlakeai/dsh-sandbox` layer disables the host `subprocess` and `fs-sandbox` providers, inserts the Tensorlake runtime, subprocess, and filesystem rows, and keeps `bash-sandbox` mounted in `danger-full-access` mode. In that mode Harness's sandbox-aware Bash executor delegates directly to the Tensorlake subprocess provider while still satisfying the permission-preset capability contract.

## Smoke test

Run one headless task that exercises both the subprocess and filesystem providers:

```sh
dsh --profile headless \
  "Use Bash to run pwd and id. Create smoke-test.txt containing hello, read it back, and report the results."
```

A successful run reports `/home/tl-user/workspace` from `pwd`, the `tl-user` identity from `id`, and reads `hello` back from the file. The model-facing working directory is the same remote Linux path, so the response should not mention or fall back from a host-machine path.

## Configuration

The bundle starts an ephemeral sandbox on profile boot and terminates it when dsh exits. The runtime module accepts these Cordis config fields:

Each run prints the sandbox ID at both lifecycle boundaries. The IDs should match:

```text
Tensorlake sandbox created: <sandbox-id>
Tensorlake sandbox terminated: <sandbox-id>
```

| Field | Default | Meaning |
|---|---:|---|
| `apiKey` | `TENSORLAKE_API_KEY` | Tensorlake API credential used only by the host SDK |
| `cwd` | `/home/tl-user/workspace` | Absolute Linux working directory shared by file and process providers |
| `timeoutSecs` | `600` | Sandbox inactivity timeout |
| `cpus` | Tensorlake default | Virtual CPU allocation |
| `memoryMb` | Tensorlake default | Memory allocation in MiB |
| `diskMb` | Tensorlake default | Root disk allocation in MiB |

The shipped bundle derives both the runtime cwd and policy workspace from `DSH_TENSORLAKE_CWD`. Prefer that single setting when changing the workspace so the Bash policy and remote providers cannot drift:

```sh
DSH_TENSORLAKE_CWD=/workspace/project dsh --profile headless "build and test this repo"
```

To configure the rows directly in the profile's `cordis.patch.yml`, override both together. A patch replaces the complete config, so restate every non-default field you need:

```yaml
- id: sandbox-policy
  config:
    mode: danger-full-access
    workspaceRoot: /workspace/project

- id: tensorlake-runtime
  config:
    cwd: /workspace/project
    timeoutSecs: 1800
    cpus: 2
    memoryMb: 4096
```

`apiKey` is optional and should normally remain omitted. The package never copies `TENSORLAKE_API_KEY`, `DEEPSEEK_API_KEY`, other credential-shaped environment variables, or `DSH_*` variables into sandbox processes. A caller may still pass an explicit environment entry through a Harness tool or service request.

## Runtime requirements

The Tensorlake image must provide `bash`, Node.js, and GNU `base64`, `cat`, `chmod`, `env`, `find`, `grep`, `ln`, `mkdir`, `mktemp`, `mv`, `ps`, `realpath`, `rm`, `stat`, and `tee`. The default managed Ubuntu image provides these tools. The runtime verifies that a configured cwd is writable and uses the managed image's passwordless `sudo` to create and hand off a protected path when necessary.

The package targets `@deepseek-ai/dsh` `0.1.0-rc.6` or later compatible release. The dsh installation supplies its optional Cordis, filesystem, subprocess, and Schemastery peers through the profile module fallback. The package uses only public `ctx.fs` and `ctx.subprocess` service definitions; no DeepSeek Harness source registration, generated catalogs, or in-repository configuration is required.

## Known limitations

- `tensorlake@0.5.103`, the current SDK release, pins `undici@8.3.0` and `nanoid@3.3.11`; `npm audit --omit=dev` reports high-severity advisories for those transitive versions. No audit-clean current Tensorlake SDK release is available, so review the upstream advisories before production use and update the SDK pin when Tensorlake publishes one.

## Develop

```sh
npm install
npm run check
npm pack
```

The three Loader entry points are `@tensorlakeai/dsh-sandbox/runtime`, `@tensorlakeai/dsh-sandbox/filesystem`, and `@tensorlakeai/dsh-sandbox/subprocess`. Each module default-exports its service class; do not add function-plugin named exports to those modules because the Cordis Loader treats mixed export forms as a function-plugin namespace.
