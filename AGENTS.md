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

### Building inside an unprivileged container

`bubblewrap` refuses to start when launched as non-root with inherited ambient
caps (`bwrap: Unexpected capabilities but not setuid, old file caps config?`),
which is the default state inside the Claude Code sandbox and similar nested
environments. Wrap the build in `unshare -r` to create a fresh user namespace
where the current uid maps to root — bwrap then sees the expected uid+caps and
proceeds normally:

```sh
unshare -r just build <pkg>.yaml
```

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

**Important:** Also add the package to the `packageRules` disable list in `renovate.json5` (the entry with `"matchDatasources": ["github-tags"]`). This prevents the github-tags manager — which matches all GitHub URLs — from also opening a duplicate bump MR. The package name must match the `depName` captured from the `repository:` URL (e.g. `intel/compute-runtime`).

## Architecture restrictions

If a package only makes sense on one arch (e.g. Intel GPU drivers are x86_64-only), add:

```yaml
package:
  target-architecture:
    - x86_64
```

Melange will then skip other arches with `nothing to build`, safely. CI reads `target-architecture` automatically (via `.gitlab/gen-dag.sh`) and skips emitting jobs for unsupported arches.

## Repackaging upstream binaries

If compiling from source is being troublesome, when upstream ships official `.deb` / `.rpm` releases (e.g. on GitHub releases) and those binaries are compatible with Wolfi's glibc, repackaging is often dramatically faster than building from source. Pattern:

```sh
ar p <pkg>.deb data.tar.gz | tar xz -C "${{targets.destdir}}"
# Relocate Debian multiarch paths (/usr/lib/x86_64-linux-gnu, /usr/local/lib) to Wolfi's /usr/lib.
```

Sanity-check compatibility first: `readelf -V <sofile> | grep GLIBC_` should show a max `GLIBC_X.Y` ≤ Wolfi's current glibc.

## CI build graph

CI uses a generated child pipeline. `.gitlab/gen-dag.sh` introspects every `*.yaml` via `melange query`, resolves build-time + runtime deps against a map of locally-provided package and subpackage names, and emits one job per `(yaml, arch)` with `needs:` set on local dependencies. Adding, renaming, or deleting a yaml needs no `.gitlab-ci.yml` edit — the graph is rebuilt every pipeline run.

## Commit Guidelines

- Create a feature branch for each change
- Format: `<package-name>: <concise description>`
- For version updates: `<package-name>/<version> package update`
- Use imperative mood ("Add feature" not "Added feature")
- Describe what changed and why, not how

## Code Style

- Format every yaml with `yam` (run `just lint`).
- 2-space indent, no trailing whitespace.
- Within `package:`, `environment:`, etc. — alphabetical fields where it doesn't break readability.
- Inside `environment.contents.packages` — alphabetical, one per line.
- Don't remove comments unless they're inaccurate; they encode why.

## Epoch Rules

melange caches builds keyed by `(version, epoch)`. If you change content but bump neither, the apk index keeps the old build and your change is silently ignored.

- **Bump `epoch:` by 1** when content changes but `version:` stays. CI's `.gitlab/epoch-check.sh` enforces this on MRs.
- **Reset `epoch: 0`** when bumping `version:`.
- **Don't bump epoch** for comment-only / pure-formatting / docs-only changes.
- Keep the trailing `# why` comment on the epoch line short and specific (`epoch: 3 # rebuild for openssl 3.5`).

## CVE Patches

- Drop the patch at `<package-name>/<CVE-ID>.patch` (e.g. `redis/CVE-2024-12345.patch`).
- Add a `patch` step to the pipeline:
  ```yaml
  - uses: patch
    with:
      patches: CVE-2024-12345.patch
  ```
- Bump `epoch:` and reference the CVE in the epoch comment.
- Prefer the upstream-published patch; second choice is the patch Alpine or Fedora applied; last choice is your own.
- If the CVE doesn't apply to how the package is built, NACK it in the commit message rather than carrying a no-op patch.

## Testing Skills

The bar: every package's `test:` section must exercise the actual functionality, not just dependencies or "loads".

### Rules

- **`set -eu`** at the top of every multi-line `runs:` block.
- **Test the actual package**, not its deps. Loading `Foo::Bar` is a smoke test, not a real test — also assert behaviour.
- **Loose version checks**: `--version` / `--help` exit-code is enough; never assert the output string format. Patch releases reformat output, suffixes get added, and the test breaks for no reason.
- **`grep -F`** for literal-string matches so `.` in version strings isn't a regex metachar.
- **Busybox-compatible shell** only: POSIX sh, no bashisms. `source` → `.`, `[[ ]]` → `[ ]`, `${var,,}` is forbidden, `[ "$x" -eq 3 ]` for numeric comparison.
- **Multi-line block scalar** for list inputs (e.g. `bins: |`), never space-separated.
- **Don't test arg-validation / error paths** — only successful invocations. Error messages change between releases.
- **For commands expected to fail**: `! cmd-that-should-fail`. Don't rely on the test runner.
- **Test environment is ephemeral**: any non-zero exit fails. Don't add `|| exit 1` after every command.

### Decision table — which test pipeline to use

| Subpackage type | Pipeline |
|---|---|
| `-doc` (manpages, info, html) | `test/docs` |
| Primary package with binaries | `test/ver-check` + `test/help-check` + at least one functional invocation |
| Primary package with shared libs | `test/ldd-check` |
| Perl module | `test/perl-module-check` + a functional assertion |
| systemd services / sockets | `test/verify-service` |
| Anything else | inline `runs:` exercising the real entry point |

### Anti-patterns to fix on sight

| Bad | Better |
|---|---|
| `pipeline: []` (no test) | At minimum `test/ver-check` or `test/perl-module-check` |
| Only `- uses: test/no-docs` | Add a real functional test alongside |
| `stat /usr/bin/foo` | `foo --version` (just exit code) plus a real invocation |
| `cmd && echo OK` (no failure propagation) | `set -eu` and let exit codes do the work |
| Asserting exact version string | Drop the assertion; exit code is enough |

## Build Environment Minimization

Build envs accumulate cruft from copy-paste. Trim aggressively.

| Often-unnecessary | Keep only when |
|---|---|
| `autoconf`, `automake`, `build-base` | The package actually compiles C/C++ |
| `ca-certificates-bundle` | The build fetches over HTTPS during compile |
| `pkgconf` | The build invokes `pkg-config` |
| `wolfi-base`, `busybox`, `apk-tools`, `wolfi-keys` | Never in test envs — already implicit |

`just lint` strips redundant test env entries automatically.

### Family rules

- **Pure-Perl modules (no XS)**: only `busybox` + `perl` (+ explicit `perl-*` deps the module declares). Drop `autoconf`/`automake`/`build-base`/`ca-certificates-bundle`.
- **Pure-Python wheels via `py/pip-build-install`**: keep `build-base` only if the wheel ships C extensions.
- **Repackaging upstream `.deb`/`.rpm`**: only `busybox` + `binutils` (for `ar`).
- **Static config-only packages** (e.g. shipping yaml/conf to /etc): only `busybox`.

## Language-Specific Skills

### Go

- Pin a Go version stream in `environment.contents.packages` (e.g. `go-1.25`), never bare `go`.
- Add `go-package: go-1.25` (matching the env entry) to **every** `go/build` and `go/bump` step. Without it, the pipeline picks whatever `go` is on `$PATH`.
- Use `go/bump` to bump indirect deps for CVEs rather than carrying a vendored patch.

### Perl

- Pure-Perl distributions need only `busybox` + `perl` at build time.
- Don't list `perl-*` runtime deps that come transitively via the dist's own `Makefile.PL`/`cpanfile`. melange's SCA picks them up.
- Ship manpages in a `-doc` subpackage via `split/manpages`.
- Test with `test/perl-module-check` plus a functional assertion (instantiate, call a method, compare output). **Always pass `modules:` as a space-separated string** (`modules: "Foo::Bar Baz::Qux"`), never a multiline block scalar — melange's shell parser rejects `for mod in\nFoo` as missing `do`.

### Python

- Use `py3.x-supported-y` packages whenever they exist.
- **Application packages built with `uv pip install`** must NOT list `py3.x-*` runtime deps — uv bundles them. List only `python-${{vars.python-version}}-base`.
- **Application packages built with `py/pip-build-install`** MUST list `py3.x-*` runtime deps — it installs the app only, deps come from apk.
- The two rules look contradictory; the difference is which build pipeline you use. Read the pipeline yaml if unsure.

### Java

- Pick a modern, supported JVM/JDK (`openjdk-21`, `openjdk-25` etc.). Avoid obsolete versions.
- Set `JAVA_HOME` and load env vars properly in tests.

### Node.js

- Install code to `/usr/lib/<package>/`.
- Wrapper script in `/usr/bin/<package>` that calls `exec node /usr/lib/<package>/...`.
- Declare `nodejs` as runtime dependency.
- Use `split/alldocs` for documentation split.
- Trust the supply-chain hardening env vars in `common.env` — they block install scripts and keep npm using system tools.

## Common Mistakes to Avoid

- Testing a dependency instead of the package being built.
- Asserting exact version strings (`grep "1.2.3"` breaks on the next patch release).
- Listing `wolfi-base` / `busybox` / `apk-tools` in test envs — already implicit.
- Forgetting to bump `epoch:` after a content change.
- Forgetting to reset `epoch: 0` when bumping `version:`.
- Bare `go` in `environment.contents.packages` instead of `go-1.X`.
- Pinning `expected-sha256` on `fetch:` URIs that renovate manages — renovate can't update hashes, so the bump MR breaks. Use `git-checkout` (which resolves SHAs naturally) or omit the hash.
- Putting `resources:` / `test-resources:` anywhere except as a direct child of `package:`.

## Search Similar Packages First

Before authoring a new yaml or refactoring an old one, find 2–3 working examples:

- This repo: `grep -l "uses: go/build" *.yaml`, `ls perl-*.yaml`, etc.
- `wolfi-os/`: `grep -l "uses: go/build" wolfi-os/*.yaml | head` — vast catalogue of patterns.
- Alpine aports: `https://gitlab.alpinelinux.org/alpine/aports/-/raw/master/<section>/<package>/APKBUILD` (sections: `main`, `community`, `testing`).

Read them before generating something from scratch.

## Slash Commands

When run with a specific yaml path as argument, these focus the agent on a single mechanical transformation:

| Command | Effect |
|---|---|
| `/build-env-minimize <pkg>.yaml` | Apply Build Environment Minimization rules |
| `/functional-tests <pkg>.yaml` | Replace placeholder/load-only tests using the decision table |
| `/go-package <pkg>.yaml` | Enforce Go version stream consistency |
| `/epoch-bump <pkg>.yaml` | Bump epoch (or reset to 0 on version bump) |
| `/apply-skills <pkg>.yaml` | Run all of the above against one package |

Each is defined under `.claude/commands/` and refers back to the rules in this file.
