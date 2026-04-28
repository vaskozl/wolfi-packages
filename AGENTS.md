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
| `just test <pkg>.yaml` | Run melange tests |
| `just clean` | Clean output |
| `just list-local` | List packages in this repo |
| `just list` | List packages in remote APK repo |
| `just withdraw <pkg>...` | Withdraw packages from all architectures |

## Parameters

Override defaults via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCH` | auto-detected | `aarch64`, `x86_64` |
| `RUNNER` | bubblewrap | `bubblewrap`, `docker`, `qemu` |
| `KEY` | `melange.rsa` | RSA signing key file |

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

## Renovate

Renovate bumps `version:` when it sees a paired `repository: https://github.com/OWNER/REPO` (or codeberg/gitlab.freedesktop.org). `uses: git-checkout` provides it naturally; otherwise add a `# repository: ...` comment. Use `curl` in downloading URLs is required and don't pin `expected-sha256` - renovate can't update hashes.

If a package **repackages upstream `.deb`/`.rpm` assets** (i.e. depends on a published GitHub Release, not just a git tag), add `# datasource: github-releases` immediately after the `# repository: ...` line:

```yaml
# Repackages Intel's upstream .deb releases rather than building from source.
# repository: https://github.com/intel/intel-graphics-compiler
# datasource: github-releases
```

Renovate will then use the `github-releases` datasource and only open a bump PR when upstream publishes a full release with assets — not for bare git tags without a release.

## Architecture restrictions

If a package only makes sense on one arch (e.g. Intel GPU drivers are x86_64-only), add:

```yaml
package:
  target-architecture:
    - x86_64
```

Melange will then skip other arches with `nothing to build`, safely. Also move the package into the `ARCH: [amd64]`-only block in `.gitlab-ci.yml` so CI doesn't waste arm64 runner time on the guaranteed skip.

## Repackaging upstream binaries

If compiling from source is being troublesome, when upstream ships official `.deb` / `.rpm` releases (e.g. on GitHub releases) and those binaries are compatible with Wolfi's glibc, repackaging is often dramatically faster than building from source. Pattern:

```sh
ar p <pkg>.deb data.tar.gz | tar xz -C "${{targets.destdir}}"
# Relocate Debian multiarch paths (/usr/lib/x86_64-linux-gnu, /usr/local/lib) to Wolfi's /usr/lib.
```

Sanity-check compatibility first: `readelf -V <sofile> | grep GLIBC_` should show a max `GLIBC_X.Y` ≤ Wolfi's current glibc.

## Keeping `.gitlab-ci.yml` in sync

Every `<pkg>.yaml` in the repo must also appear in the `parallel.matrix` in `.gitlab-ci.yml` — otherwise CI never builds it. When you add, rename, or delete a yaml, edit the matrix in the same change. Arch-restricted packages go under the `ARCH: [amd64]` (or `[arm64]`) entry; general packages under the both-arch entry.

## Commit Guidelines

- Create a feature branch for each change
- Format: `<package-name>: <concise description>`
- For version updates: `<package-name>/<version> package update`
- Use imperative mood ("Add feature" not "Added feature")
- Describe what changed and why, not how
