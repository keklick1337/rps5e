# RustPS5Emu (RPS5E)
# Copyright (C) 2025-2026 Vladislav Tislenko aka keklick1337
# SPDX-License-Identifier: GPL-2.0-only
#
# Common tasks: build, tests, install, release packages, statistics.
# GAMES_DIR for the tests that read real games:
# local/games.conf (not in git; see scripts/games.conf.example).
CARGO ?= cargo
ARGS  ?=
PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin
DATADIR = $(PREFIX)/share
ICON_SIZES = 16 24 32 48 64 128 256 512

.PHONY: all build test test-gpu gui install uninstall run imports stats package clean help

all: build

help:
	@echo "make build | test | test-gpu | gui | install [PREFIX=~/.local] | uninstall"
	@echo "make package              (release archives in dist/: Linux tarball; Windows zip with TARGET=windows)"
	@echo "make stats                (progress/progress.svg and progress/PROGRESS_STATS.md)"
	@echo "make run                  (launcher; in game F1 opens the in-game menu)"
	@echo "make run GAME=<folder|eboot.bin> [ARGS=...]"
	@echo "ARGS examples: --input-mode keyboard  --bind Cross=KEY:Space  --fullscreen"
	@echo "Data: DATA_DIR=<dir> (saves/settings), CONFIG_DIR=<dir> (input mapping)"

build:
	$(CARGO) build --release

# GAMES_DIR of local/games.conf reaches the tests that read real games.
test:
	set -a; [ -f local/games.conf ] && . ./local/games.conf; set +a; $(CARGO) test --release

# Device tests on the real Vulkan GPU with validation layers.
test-gpu:
	VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation $(CARGO) test --release -p ps5gpu -- --ignored

# The launcher: `rps5e` without arguments (the same program as `rps5e-cli`).
gui: build
	./target/release/rps5e

# rps5e and rps5e-cli into $(BINDIR), the desktop entry and the icons into
# $(DATADIR) (applications/, icons/hicolor/).
install: build
	install -Dm755 target/release/rps5e "$(BINDIR)/rps5e"
	install -Dm755 target/release/rps5e-cli "$(BINDIR)/rps5e-cli"
	install -Dm644 packaging/rps5e.desktop "$(DATADIR)/applications/rps5e.desktop"
	for s in $(ICON_SIZES); do \
		install -Dm644 assets/icons/rps5e-$$s.png "$(DATADIR)/icons/hicolor/$${s}x$${s}/apps/rps5e.png"; \
	done
	install -Dm644 assets/rps5e-logo.svg "$(DATADIR)/icons/hicolor/scalable/apps/rps5e.svg"
	-command -v update-desktop-database >/dev/null && update-desktop-database "$(DATADIR)/applications"
	-command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$(DATADIR)/icons/hicolor"

uninstall:
	rm -f "$(BINDIR)/rps5e" "$(BINDIR)/rps5e-cli" "$(DATADIR)/applications/rps5e.desktop"
	rm -f "$(DATADIR)/icons/hicolor/scalable/apps/rps5e.svg"
	for s in $(ICON_SIZES); do rm -f "$(DATADIR)/icons/hicolor/$${s}x$${s}/apps/rps5e.png"; done

# Without GAME: the launcher (game library, settings, key bindings).
run: build
	@if [ -z "$(GAME)" ]; then ./target/release/rps5e; else BUILD=0 scripts/run-game.sh "$(GAME)" $(ARGS); fi

imports: build
	@test -n "$(GAME)" || (echo "usage: make imports GAME=<eboot.bin>"; exit 2)
	./target/release/rps5e-cli imports "$(GAME)"

# Implementation statistics: firmware functions with HLE, game imports,
# decoded shader opcodes (progress/PROGRESS_STATS.md, progress/progress.svg
# shown in README.md). Games: GAMES_DIR or local/games.conf.
stats: build
	python3 -I progress/progress.py

# Release archives (scripts/package.sh): TARGET=linux (default) or windows.
package:
	scripts/package.sh $(TARGET)

clean:
	$(CARGO) clean

