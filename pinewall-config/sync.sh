#!/usr/bin/env bash
set -euo pipefail

host="${1:-pinewall}"

# walk files safely (handles spaces/newlines) and copy each file from remote host
# skipping the specific problematic file
find vendor -type f -print0 | while IFS= read -r -d '' f; do
  if [ "$f" = "vendor/etc/systemd/network/30-wireguard.netdev" ]; then
    continue
  fi

  # strip leading "vendor/" for remote path
  remote_path="/${f#vendor/}"

  # make sure local directory exists before copying
  mkdir -p "$(dirname "./$f")"

  echo "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r ${host}:${remote_path} ./${f}"
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r "${host}:${remote_path}" "./${f}" || true
done

