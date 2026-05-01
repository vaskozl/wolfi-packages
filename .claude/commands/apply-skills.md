# apply-skills

Run all the per-package skills from `AGENTS.md` against the yaml passed as
`$ARGUMENTS`. This is the skillup-equivalent for this repo.

## Steps

1. Read `AGENTS.md` end-to-end.
2. Read `$ARGUMENTS`.
3. Apply, in order:
   - **Build Environment Minimization** (drop unused build deps per family rule)
   - **Language-Specific Skills** (Go version stream, Perl pure-Perl, Python
     uv-vs-pip-build-install, Node wrapper layout, etc.)
   - **Testing Skills** (decision table; replace anti-patterns)
   - **Code Style** (alphabetical packages, comments preserved)
4. Bump `epoch:` once per *Epoch Rules* — even if multiple skills touched the file.
5. Run `just lint $ARGUMENTS`.
6. Report a short summary: which skills changed what, and which were no-ops.
   Don't churn for the sake of it — if the file was already correct, exit
   cleanly and say so.

If you're unsure whether a transformation applies, **err on the side of not
changing**. False fixes cost more than missed ones.
