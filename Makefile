.PHONY: dev build clean local-build local-clean

CONTAINER_ENGINE ?= docker
REPO_NAME ?= ht-dv-sim
DEV_IMAGE ?= ht-dv-sim:latest
CONTAINERFILE ?= Containerfile
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

build:
	@echo "==> Building dev image..."
	@$(CONTAINER_ENGINE) build -t $(DEV_IMAGE) -f $(CONTAINERFILE) .

dev: build
	@echo "==> Entering workspace..."
	@$(CONTAINER_ENGINE) run \
	  --rm \
	  --tty \
	  --interactive \
	  --volume=./:/workspace/:rw \
	  --workdir=/workspace \
	  --name=$(REPO_NAME)-dev \
	  --user=$(HOST_UID):$(HOST_GID) \
	  --userns=host \
	  --network=host \
	  --pid=host \
	  --ipc=host \
	  -e QT_X11_NO_MITSHM=1 \
	  -e NVIDIA_DRIVER_CAPABILITIES=all \
	  -e DISPLAY=$(DISPLAY) \
	  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
	  $(DEV_IMAGE) \
	  bash

clean:
	@$(CONTAINER_ENGINE) rm -f $(REPO_NAME)-dev 2>/dev/null || true

local-build:
	@echo "==> Building locally..."
	@rm -rf build
	@cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@ln -sf $(PWD)/build/compile_commands.json compile_commands.json
	@cmake --build build

local-clean:
	@rm -rf build compile_commands.json
