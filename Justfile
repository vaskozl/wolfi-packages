# Justfile for building packages with melange and managing the APK repository

# Build variables
arch := `uname -m | sed 's/arm64/aarch64/'`
runner := env_var_or_default("RUNNER", "bubblewrap")
out_dir := justfile_directory() / "packages"
key := env_var_or_default("KEY", "melange.rsa")
arches := "x86_64 aarch64"

# Reproducibility: derive SOURCE_DATE_EPOCH from the latest commit timestamp so
# every rebuild from the same git ref produces byte-identical output. Override
# with `SOURCE_DATE_EPOCH=...` if needed (e.g. building from a dirty worktree).
export SOURCE_DATE_EPOCH := env_var_or_default("SOURCE_DATE_EPOCH", `git -C "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" log -1 --pretty=%ct 2>/dev/null || echo 0`)

# Set LINT=yes to fail builds on melange lint warnings (catches drift early).
lint_flag := if env_var_or_default("LINT", "no") == "yes" { "--fail-on-lint-warning" } else { "" }

# Use `unshare -r` to run in a user namespace if available (avoids needing setuid bwrap).
unshare := `command -v unshare >/dev/null 2>&1 && echo "unshare -r" || echo ""`

# Show available recipes
default:
    @just --list

# Regenerate RSA signing keys and remove stale package indexes
keygen:
    melange keygen
    find {{out_dir}} -name 'APKINDEX.tar.gz' -delete

# Setup RSA signing key if missing
setup:
    #!/usr/bin/env sh
    if [ ! -f {{key}} ]; then
        echo "RSA key file {{key}} not found. Generating..."
        just keygen
    fi
    echo "RSA key setup complete"

# Build a specific package (e.g. just build redis.yaml)
build pkg: setup
    #!/usr/bin/env sh
    echo "Building {{pkg}} for {{arch}}..."
    mkdir -p {{out_dir}}
    {{unshare}} melange build "{{pkg}}" \
        --workspace-dir="{{justfile_directory()}}" \
        --arch="{{arch}}" \
        --runner={{runner}} \
        --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,{{out_dir}} \
        --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
        --signing-key={{key}} \
        --pipeline-dirs=./pipelines \
        --env-file=common.env \
        --source-dir="$(basename {{pkg}} .yaml)" \
        --out-dir="{{out_dir}}" \
        {{lint_flag}}

# Build all packages
build-all: setup
    #!/usr/bin/env sh
    echo "Building all packages for {{arch}}..."
    mkdir -p {{out_dir}}
    for pkg in *.yaml; do
        echo "Building $pkg for {{arch}}..."
        just build "$pkg" || exit 1
    done

# Test a specific package (e.g. just test redis.yaml)
test pkg: setup
    #!/usr/bin/env sh
    echo "Testing {{pkg}} for {{arch}}..."
    mkdir -p {{out_dir}}/{{arch}}
    ln -sfn {{out_dir}}/{{arch}} {{justfile_directory()}}/{{arch}}
    {{unshare}} melange test "{{pkg}}" \
        --workspace-dir="{{justfile_directory()}}" \
        --arch="{{arch}}" \
        --runner={{runner}} \
        --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,{{out_dir}} \
        --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
        --pipeline-dirs ./pipelines \
        --env-file=common.env \
        --test-package-append=busybox \
        --ignore-signatures
    rm -f {{justfile_directory()}}/{{arch}}

# Lint yamls: format with yam + structural checks. No args = all *.yaml.
lint *files:
    #!/usr/bin/env bash
    set -euo pipefail

    yamls=({{files}})
    if [ ${#yamls[@]} -eq 0 ]; then
      shopt -s nullglob
      yamls=(*.yaml)
    fi

    have_yq=true; command -v yq >/dev/null 2>&1 || have_yq=false
    have_yam=true; command -v yam >/dev/null 2>&1 || have_yam=false
    [ "$have_yq" = "true" ]  || echo "WARN: yq not installed — skipping structural checks"
    [ "$have_yam" = "true" ] || echo "WARN: yam not installed — skipping format pass"

    fails=0
    for fn in "${yamls[@]}"; do
      case "$fn" in
        *.yaml) ;;
        *) echo "--- $fn not a yaml file, skipping"; continue ;;
      esac

      if [ "$have_yq" = "true" ]; then
        pkg=$(yq -r '.package.name' "$fn")
        if [ -z "$pkg" ] || [ "$pkg" = "null" ]; then
          echo "FAIL [$fn]: no package.name"
          fails=$((fails+1)); continue
        fi
        echo "--- $pkg"

        # Strip redundant test env deps (already implicit in wolfi-base/test).
        for redundant in wolfi-base busybox apk-tools wolfi-keys "$pkg"; do
          yq -i 'del(.test.environment.contents.packages[] | select(. == "'"$redundant"'"))' "$fn"
        done

        # Drop empty test.environment.contents.
        n=$(yq -r '.test.environment.contents.packages // [] | length' "$fn")
        if [ "$n" = "0" ]; then
          yq -i 'del(.test.environment.contents)' "$fn"
        fi

        # resources:/test-resources: must be a direct child of `package:`.
        bad=$(yq '[.. | path] | .[] | select(length > 0 and (.[-1] == "resources" or .[-1] == "test-resources")) | join(".")' "$fn" \
          | grep -vxE 'package\.(resources|test-resources)' || true)
        if [ -n "$bad" ]; then
          echo "FAIL [$fn]: 'resources:'/'test-resources:' must be a direct child of 'package:'. Found at:"
          echo "$bad" | sed 's/^/  /'
          fails=$((fails+1)); continue
        fi
      else
        echo "--- $fn"
      fi

      if [ "$have_yam" = "true" ]; then
        yam "$fn" || { echo "FAIL [$fn]: yam"; fails=$((fails+1)); }
      fi
    done

    if [ "$fails" -gt 0 ]; then
      echo ""
      echo "$fails file(s) failed lint"
      exit 1
    fi

# Clean build output
clean:
    rm -rf {{out_dir}}

# List packages defined in this repo
list-local:
    @for pkg in *.yaml; do echo "$pkg"; done

# Run wolfictl in Docker against the packages/ output directory
_wolfictl *args:
    docker run -i --rm -w {{out_dir}} -v {{out_dir}}:{{out_dir}} -v ~/melange.rsa:/root/melange.rsa:ro ghcr.io/vaskozl/wolfictl {{args}}

# List packages in the remote APK repository
list:
    just _wolfictl apk ls https://apks.sko.ai/x86_64/APKINDEX.tar.gz

# Withdraw one or more packages from all architectures
withdraw +packages:
    #!/usr/bin/env sh
    for arch in {{arches}}; do
        echo "Withdrawing {{packages}} from $arch..."
        if (cd {{out_dir}}/$arch && just --justfile "{{justfile()}}" _wolfictl withdraw --signing-key ~/melange.rsa {{packages}} < APKINDEX.tar.gz > APKINDEX.tar.gz.new); then
            chmod 644 {{out_dir}}/$arch/APKINDEX.tar.gz.new
            mv {{out_dir}}/$arch/APKINDEX.tar.gz {{out_dir}}/$arch/APKINDEX.tar.gz.bak
            mv {{out_dir}}/$arch/APKINDEX.tar.gz.new {{out_dir}}/$arch/APKINDEX.tar.gz
            (cd {{out_dir}}/$arch && rm -f {{packages}})
            echo "✓ Successfully withdrew from $arch and deleted packages"
        else
            echo "✗ Failed to withdraw from $arch"
            rm -f {{out_dir}}/$arch/APKINDEX.tar.gz.new
            exit 1
        fi
    done
    echo "Done!"
