# Building Packages

```sh
make <package>.yaml
# Example:
make pinewall-config.yaml
```

Built packages appear in `packages/aarch64/`.

## Testing Packages

```sh
make test PKG=pinewall-config.yaml
```

The test runs in a bubblewrap container, installs the package, and verifies:
- Config files exist (`/etc/hostname`, `/etc/hosts`, `/etc/nftables.nft`, `/etc/blocky.yaml`)
- Command `/usr/bin/set-irq-affinity` is executable

### Manual Install Test (Alternative)

```sh
# Build and install
make pinewall-config.yaml
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

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make <pkg>.yaml` | Build package |
| `make build PKG=<file>` | Build package |
| `make build-all` | Build all packages |
| `make test PKG=<file>` | Run melange tests |
| `make clean` | Clean output |
| `make list` | List packages |

## Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCH` | auto | `aarch64`, `amd64` |
| `RUNNER` | bubblewrap | `bubblewrap`, `docker`, `qemu` |
| `OUT_DIR` | `./packages` | Output directory |

## Node.js Packages

- Install to `/usr/lib/<package>/`
- Wrapper in `/usr/bin/<package>` with `exec node ...`
- Declare `nodejs` runtime dependency
- Use `split/alldocs` for docs subpackage
