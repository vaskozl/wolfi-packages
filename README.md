# wolfi-packages

Custom [`melange`](https://github.com/chainguard-dev/melange) recipes for packages that aren't in Wolfi yet.

`melange` is the `.deb`/`.rpm` model applied to Wolfi: controlled build sandbox, declared dependencies, a defined set of files in the output APK. Everything here is either something Wolfi doesn't ship (niri, bootc, linux-lts, ostree, composefs, dracut, asahi-scripts) or something that is Wolfi-compatible config packaged as an APK (nori-user, pinewall-config) to avoid ad-hoc `RUN useradd` and friends.

Renovate keeps `version:` pins fresh automatically — it needs a paired `repository:` comment or `git-checkout` step to know what to watch.

## Build

```sh
just build niri.yaml
```

Built APKs land in `packages/aarch64/` or `packages/x86_64/`. The recipes auto-wrap `melange` in `unshare -r` when available, so builds work unchanged inside the Claude Code sandbox or any other unprivileged container.

Run `just` with no args to list all recipes.

## Test

```sh
just test niri.yaml
```

Tests run in a bubblewrap container, install the package from the local index, and exercise the actual binary — not just `stat /usr/bin/foo`.

## Lint

```sh
just lint
```

Formats every yaml with `yam`, drops redundant test env deps, and checks structural invariants. CI runs this on every MR.

## Related

- apkontainers (`apko` configs that consume these APKs): [`../`](../)
- Router config tree: [`vaskozl/pinewall-config`](https://github.com/vaskozl/pinewall-config)
- Bluefin's prior bootc-on-Wolfi work: [`projectbluefin/wolfifin`](https://github.com/projectbluefin/wolfifin)
