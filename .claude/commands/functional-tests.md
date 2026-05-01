# functional-tests

Apply the **Testing Skills** rules from `AGENTS.md` to the yaml passed as
`$ARGUMENTS`.

## Steps

1. Read `AGENTS.md` (sections: *Testing Skills*, *Language-Specific Skills*).
2. Read `$ARGUMENTS`.
3. Identify package type (CLI binaries, library, daemon, perl module, config-only).
4. Look up the right test pipeline in the *Decision table* and apply it.
5. Replace any anti-patterns from the *Anti-patterns to fix on sight* table.
6. Bump `epoch:` per *Epoch Rules*.
7. Run `just lint $ARGUMENTS`.
8. If possible, validate locally:
   ```sh
   unshare -r just build $ARGUMENTS
   unshare -r just test  $ARGUMENTS
   ```
9. Skip cleanly if there's genuinely nothing to test (data-only packages).
