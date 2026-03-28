# Building Packages

```sh
just build <package>.yaml
# Example:
just build pinewall-config.yaml
```

Built packages appear in `packages/aarch64/`.

## Testing Packages

```sh
just test pinewall-config.yaml
```

The test runs in a bubblewrap container, installs the package, and verifies:
- Config files exist (`/etc/hostname`, `/etc/hosts`, `/etc/nftables.nft`, `/etc/blocky.yaml`)
- Command `/usr/bin/set-irq-affinity` is executable

### Manual Install Test (Alternative)

```sh
# Build and install
just build pinewall-config.yaml
apk add --allow-untrusted packages/aarch64/pinewall-config-*.apk

# Verify
apk info -L pinewall-config

# Cleanup
apk del pinewall-config
```

## Package Test Example

The `test:` section in `pinewall-config.yaml`:

```yaml
test:
  pipeline:
    - runs: |
        stat /etc/hostname
        stat /etc/hosts
        stat /etc/nftables.nft
        stat /etc/blocky.yaml
        /usr/bin/set-irq-affinity --help || true
```

## Justfile Recipes

| Recipe | Description |
|--------|-------------|
| `just build <pkg>.yaml` | Build a specific package |
| `just build-all` | Build all packages |
| `just test <pkg>.yaml` | Run melange tests for a package |
| `just clean` | Clean output directory |
| `just list` | List available package YAML files |
| `just index-list` | List packages in the remote index |
| `just withdraw <pkg>...` | Withdraw packages from all architecture indexes |

## Parameters

Override defaults by setting variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `arch` | auto-detected | `aarch64`, `x86_64` |
| `runner` | `bubblewrap` | `bubblewrap`, `docker`, `qemu` |
| `out_dir` | `./packages` | Output directory |

Example: `just arch=x86_64 runner=docker build redis.yaml`

## Reference Packages

- **This repo**: Custom package examples - use as primary reference
- **wolfi-os**: https://github.com/wolfi-dev/os - melange patterns
- **Alpine aports**: https://gitlab.alpinelinux.org/alpine/aports - search for APKBUILD files

### Using Alpine APKBUILD as Reference

When packaging software, check Alpine's aports for existing build recipes:

```
https://gitlab.alpinelinux.org/alpine/aports/-/raw/master/<section>/<package>/APKBUILD
```

Sections to check (in order): `main`, `community`, `testing`

**Example:** https://gitlab.alpinelinux.org/alpine/aports/-/raw/master/community/maddy/APKBUILD

**Important differences:**
- Wolfi uses `glibc` instead of `musl`
- Wolfi uses **systemd** (`-system` suffix) - Alpine uses **OpenRC** (`-openrc` suffix)
- Only create systemd service packages if the upstream repository provides service files

## Node.js Packages

- Install code to `/usr/lib/<package>/`
- Create wrapper script in `/usr/bin/<package>` that calls `exec node /usr/lib/<package>/...`
- Declare `nodejs` as runtime dependency
- Use `split/alldocs` for documentation split

## General Guidelines

- Search for similar packages to use as working examples before creating new ones
- Split documentation into a separate subpackage where possible
- Reuse existing melange pipelines where possible
- When debugging failed builds, clone the source code locally to study the build system
- Explore APK contents: `tar tzv -f packages/aarch64/<package>-*.apk`

## Commit Guidelines

- Create a feature branch for each change
- Format: `<package-name>: <concise description>`
- For version updates: `<package-name>/<version> package update`
- Use imperative mood ("Add feature" not "Added feature")
- Describe what changed and why, not how
