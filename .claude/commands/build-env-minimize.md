# build-env-minimize

Apply the **Build Environment Minimization** rules from `AGENTS.md` to the
yaml passed as `$ARGUMENTS`.

## Steps

1. Read `AGENTS.md` (sections: *Build Environment Minimization*, *Language-Specific Skills*).
2. Read `$ARGUMENTS`.
3. Identify the build family from the pipeline (perl/make? go/build? autoconf?
   pure copy? repackage `.deb`?).
4. Remove unused entries from `environment.contents.packages` per the family rule.
5. Bump `epoch:` per *Epoch Rules* (cached build is invalidated).
6. Run `just lint $ARGUMENTS`.
7. Report what was removed and which family rule applied. If the package was
   already minimal, say so and exit without changes.
