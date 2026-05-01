# epoch-bump

Apply **Epoch Rules** from `AGENTS.md` to the yaml passed as `$ARGUMENTS`.

## Steps

1. Read `AGENTS.md` → *Epoch Rules*.
2. Read `$ARGUMENTS`.
3. If `version:` was bumped vs `git show main:$ARGUMENTS`, set `epoch: 0`.
4. Otherwise increment `epoch:` by 1.
5. Update the trailing comment on the epoch line if present (short, *why*).
6. Run `just lint $ARGUMENTS`.
