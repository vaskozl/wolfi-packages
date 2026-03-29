# Justfile for building packages with melange and managing the APK repository

# Build variables
arch := `uname -m | sed 's/arm64/aarch64/'`
runner := env_var_or_default("RUNNER", "bubblewrap")
out_dir := justfile_directory() / "packages"
key := env_var_or_default("KEY", "melange.rsa")
arches := "x86_64 aarch64"

# Show available recipes
default:
    @just --list

# Setup RSA signing key if missing
setup:
    #!/usr/bin/env sh
    if [ ! -f {{key}} ]; then
        echo "RSA key file {{key}} not found. Generating..."
        melange keygen
    fi
    echo "RSA key setup complete"

# Build a specific package (e.g. just build redis.yaml)
build pkg: setup
    #!/usr/bin/env sh
    echo "Building {{pkg}} for {{arch}}..."
    mkdir -p {{out_dir}}
    melange build "{{pkg}}" \
        --workspace-dir="{{justfile_directory()}}" \
        --arch="{{arch}}" \
        --runner={{runner}} \
        --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,{{out_dir}} \
        --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
        --signing-key={{key}} \
        --pipeline-dir=./pipelines \
        --env-file=common.env \
        --source-dir="$(basename {{pkg}} .yaml)" \
        --out-dir="{{out_dir}}"

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
    melange test "{{pkg}}" \
        --workspace-dir="{{justfile_directory()}}" \
        --arch="{{arch}}" \
        --runner={{runner}} \
        --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,{{out_dir}} \
        --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
        --env-file=common.env \
        --test-package-append=busybox \
        --ignore-signatures
    rm -f {{justfile_directory()}}/{{arch}}

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
