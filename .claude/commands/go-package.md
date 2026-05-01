# go-package

Enforce **Go version stream consistency** (see `AGENTS.md` →
*Language-Specific Skills* → *Go*) on the yaml passed as `$ARGUMENTS`.

## Steps

1. Read `AGENTS.md`.
2. Read `$ARGUMENTS`. Exit if no `uses: go/build` / `uses: go/bump`.
3. Find the `go-*` entry in `environment.contents.packages`. If bare `go`,
   replace with the latest stream — pick from
   `grep -hE '^      - go-[0-9.]+$' wolfi-os/*.yaml | sort -u | tail -1`.
4. Add `go-package: go-X.Y` (matching the env entry) to every `go/build` and
   `go/bump` step that lacks it.
5. Bump `epoch:`.
6. Run `just lint $ARGUMENTS`.
