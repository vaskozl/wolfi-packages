# Justfile for building packages with melange and wolfictl operations

# Default values
arch := `uname -m | sed 's/arm64/aarch64/'`
out_dir := justfile_directory() / "packages"
workspace_dir := justfile_directory()
runner := "bubblewrap"
key := "melange.rsa"

arches := "x86_64 aarch64"

melange_flags := "--workspace-dir=" + workspace_dir + \
    " --arch=" + arch + \
    " --runner=" + runner + \
    " --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,./packages" + \
    " --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub" + \
    " --signing-key=" + key + \
    " --pipeline-dir=./pipelines" + \
    " --env-file=common.env"

# Show available recipes
help:
    @just --list

# Setup RSA signing key (generate if not present)
setup:
    @if [ ! -f {{ key }} ]; then \
        echo "RSA key file {{ key }} not found. Generating..."; \
        melange keygen; \
    fi
    @echo "RSA key setup complete"

# Build a specific package: just build redis.yaml
build pkg: setup
    @echo "Building {{ pkg }} for {{ arch }}..."
    @mkdir -p {{ out_dir }}
    melange build "{{ pkg }}" {{ melange_flags }} --source-dir="{{ without_extension(pkg) }}" --out-dir="{{ out_dir }}" --signing-key={{ key }}

# Build all packages
build-all: setup
    #!/usr/bin/env sh
    set -e
    mkdir -p {{ out_dir }}
    for pkg in *.yaml; do
        echo "Building $pkg for {{ arch }}..."
        melange build "$pkg" {{ melange_flags }} --source-dir="$(basename "$pkg" .yaml)" --out-dir="{{ out_dir }}" --signing-key={{ key }}
    done

# Test a specific package: just test redis.yaml
test pkg: setup
    @echo "Testing {{ pkg }} for {{ arch }}..."
    @mkdir -p {{ out_dir }}/{{ arch }}
    @ln -sfn {{ out_dir }}/{{ arch }} {{ workspace_dir }}/{{ arch }}
    melange test "{{ pkg }}" \
        --workspace-dir="{{ workspace_dir }}" \
        --arch="{{ arch }}" \
        --runner={{ runner }} \
        --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os,{{ out_dir }} \
        --keyring-append=melange.rsa.pub,https://apks.sko.ai/melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
        --env-file=common.env \
        --test-package-append=busybox \
        --ignore-signatures
    @rm -f {{ workspace_dir }}/{{ arch }}

# Clean output directory
clean:
    @echo "Cleaning {{ out_dir }}..."
    rm -rf {{ out_dir }}

# List available package YAML files
list:
    @echo "Available packages:"; for pkg in *.yaml; do echo "  $pkg"; done

# --- wolfictl / index management ---

# Run wolfictl command in Docker
wolfictl *args:
    docker run -i --rm -w {{ justfile_directory() }} -v {{ justfile_directory() }}:{{ justfile_directory() }} -v ~/melange.rsa:/root/melange.rsa:ro ghcr.io/vaskozl/wolfictl {{ args }}

# List packages in the remote index
index-list:
    just wolfictl apk ls https://apks.sko.ai/x86_64/APKINDEX.tar.gz

# Withdraw one or more packages from all architectures: just withdraw "pkg1 pkg2"
withdraw +packages:
    #!/usr/bin/env sh
    set -e
    for arch_dir in {{ arches }}; do
        echo "Withdrawing {{ packages }} from $arch_dir..."
        if (cd "$arch_dir" && just wolfictl withdraw --signing-key ~/melange.rsa {{ packages }} < APKINDEX.tar.gz > APKINDEX.tar.gz.new); then
            chmod 644 "$arch_dir/APKINDEX.tar.gz.new"
            mv "$arch_dir/APKINDEX.tar.gz" "$arch_dir/APKINDEX.tar.gz.bak"
            mv "$arch_dir/APKINDEX.tar.gz.new" "$arch_dir/APKINDEX.tar.gz"
            (cd "$arch_dir" && rm -f {{ packages }})
            echo "Successfully withdrew from $arch_dir and deleted packages"
        else
            echo "Failed to withdraw from $arch_dir"
            rm -f "$arch_dir/APKINDEX.tar.gz.new"
            exit 1
        fi
    done
    echo "Done!"
