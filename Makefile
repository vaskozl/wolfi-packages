# Makefile for building packages with melange
# Based on .gitlab-ci.yml configuration

# Default values
ARCH ?= $(shell uname -m)
ifeq (${ARCH}, arm64)
	ARCH = aarch64
endif
PKG ?=
OUT_DIR ?= ./packages
WORKSPACE_DIR ?= $(PWD)
RUNNER ?= docker
KEY ?= melange.rsa

# Package list - automatically discover all YAML files
PACKAGES := $(wildcard *.yaml)

# Melange flags from gitlab-ci.yml
MELANGE_FLAGS := --workspace-dir="$(WORKSPACE_DIR)" \
                --arch="$(ARCH)" \
                --runner=$(RUNNER) \
                --repository-append=https://apks.sko.ai,https://packages.wolfi.dev/os \
                --keyring-append melange.rsa.pub,https://packages.wolfi.dev/os/wolfi-signing.rsa.pub \
				--pipeline-dir ./pipelines \
                --env-file common.env

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build PKG=<package.yaml> [ARCH=<arch>] [RUNNER=<runner>]  - Build a specific package"
	@echo "  build-all [ARCH=<arch>] [RUNNER=<runner>]                 - Build all packages"
	@echo "  clean                                                     - Clean output directory"
	@echo "  setup                                                     - Setup RSA key (if needed)"
	@echo "  list                                                      - List available packages"
	@echo ""
	@echo "Parameters:"
	@echo "  ARCH=<arch>      - Target architecture (default: auto-detected)"
	@echo "  RUNNER=<runner>  - Container runner: bubblewrap, docker, qemu (default: bubblewrap)"
	@echo "  KEY=<keyfile>    - RSA signing key file (default: melange.rsa)"
	@echo ""
	@echo "Examples:"
	@echo "  make build PKG=redis.yaml"
	@echo "  make redis.yaml                        # Shorthand for above"
	@echo "  make build PKG=redis.yaml ARCH=arm64"
	@echo "  make build-all RUNNER=bubblewrap"
	@echo "  make build-all ARCH=arm64 RUNNER=docker"

# Setup RSA key (equivalent to before_script)
.PHONY: setup
setup:
	@if [ ! -f $(KEY) ]; then \
		echo "RSA key file $(KEY) not found. Please ensure it exists."; \
		exit 1; \
	fi
	@echo "RSA key setup complete"

# Build a specific package
.PHONY: build
build: setup
	@if [ -z "$(PKG)" ]; then \
		echo "Error: PKG variable must be set. Example: make build PKG=redis.yaml"; \
		exit 1; \
	fi
	@echo "Building $(PKG) for $(ARCH)..."
	@mkdir -p $(OUT_DIR)
	@melange build "$(PKG)" $(MELANGE_FLAGS) --source-dir="$(basename $(PKG))" --out-dir="$(OUT_DIR)" --signing-key=$(KEY)

# Build all packages
.PHONY: build-all
build-all: setup
	@echo "Building all packages for $(ARCH)..."
	@mkdir -p $(OUT_DIR)
	@for pkg in $(PACKAGES); do \
		echo "Building $$pkg for $(ARCH)..."; \
		melange build "$$pkg" $(MELANGE_FLAGS) --source-dir="$$(basename $$pkg .yaml)" --out-dir="$(OUT_DIR)" --signing-key=$(KEY) || exit 1; \
	done

# Build for multiple architectures
.PHONY: build-multi-arch
build-multi-arch:
	@$(MAKE) build-all ARCH=amd64
	@$(MAKE) build-all ARCH=arm64

# Test a package (commented out in original gitlab-ci.yml)
.PHONY: test
test: setup
	@if [ -z "$(PKG)" ]; then \
		echo "Error: PKG variable must be set. Example: make test PKG=redis.yaml"; \
		exit 1; \
	fi
	@echo "Testing $(PKG) for $(ARCH)..."
	@melange test "$(PKG)" $(MELANGE_FLAGS) --test-package-append=busybox

# Clean output directory
.PHONY: clean
clean:
	@echo "Cleaning $(OUT_DIR)..."
	@rm -rf $(OUT_DIR)

# List available packages
.PHONY: list
list:
	@echo "Available packages:"
	@for pkg in $(PACKAGES); do \
		echo "  $$pkg"; \
	done

# Individual package targets for convenience
.PHONY: $(PACKAGES)
$(PACKAGES): setup
	@$(MAKE) build PKG=$@
