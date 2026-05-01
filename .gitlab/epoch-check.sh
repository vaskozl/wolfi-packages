#!/usr/bin/env bash
# epoch-bot equivalent: any *.yaml whose content changed (vs the main branch)
# must have either a `version:` bump or an `epoch:` bump. Otherwise the cached
# apk won't be rebuilt and the change silently has no effect.
#
# Run in CI on merge requests against main. Skip on main itself.

set -euo pipefail

base_ref="${BASE_REF:-origin/main}"

mapfile -t changed < <(git diff --name-only "${base_ref}"...HEAD -- '*.yaml' | grep -v '^pipelines/' || true)

if [ "${#changed[@]}" -eq 0 ]; then
  echo "epoch-check: no top-level yaml changes vs ${base_ref}"
  exit 0
fi

fails=0
for fn in "${changed[@]}"; do
  [ -f "$fn" ] || continue  # deleted files are fine

  before_ver=$(git show "${base_ref}:${fn}" 2>/dev/null | yq -r '.package.version // ""' || true)
  before_ep=$(git show "${base_ref}:${fn}" 2>/dev/null | yq -r '.package.epoch // ""' || true)
  after_ver=$(yq -r '.package.version // ""' "$fn")
  after_ep=$(yq -r '.package.epoch // ""' "$fn")

  if [ "$before_ver" = "$after_ver" ] && [ "$before_ep" = "$after_ep" ]; then
    echo "FAIL: $fn changed but version ($after_ver) and epoch ($after_ep) are unchanged."
    fails=$((fails+1))
  else
    echo "OK:   $fn  version=$before_ver→$after_ver  epoch=$before_ep→$after_ep"
  fi
done

if [ "$fails" -gt 0 ]; then
  echo ""
  echo "Bump epoch (or version) on the failing yamls so apks are rebuilt."
  echo "Reset epoch to 0 when bumping version."
  exit 1
fi
