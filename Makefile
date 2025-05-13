# ===========================
#  Makefile — bigram_sampler
#
#  one-liner UX:
#      make            # → release build  (fast, optimised)
#      make debug      # → debug build   (ASan + symbols)
#      make clean      # → wipe build/
# ===========================

# ── toolchain ──────────────────────────────────────────────────────
CC      ?= gcc                   # override with CC=clang if desired
SRC     := bigram_streamer.c # <— our new optimised file
NAME    := bigram_streamer
BUILD   := build

# ── common flags ──────────────────────────────────────────────────
CFLAGS.common := -std=c11 -Wall -Wextra -Wshadow -pedantic -pipe
LDFLAGS       := -lm -pthread     # pthread now in both profiles

# ── release (speed) ───────────────────────────────────────────────
CFLAGS.release := $(CFLAGS.common) -Ofast -march=native -funroll-loops \
                  -fomit-frame-pointer -flto
TARGET.release := $(BUILD)/$(NAME)_release

# ── debug / ASan (safety) ─────────────────────────────────────────
CFLAGS.debug  := $(CFLAGS.common) -O0 -g3 -fsanitize=address,undefined \
                 -fno-omit-frame-pointer
TARGET.debug  := $(BUILD)/$(NAME)_debug

# ── convenience targets ───────────────────────────────────────────
.PHONY: all release debug clean

all: release          ## default: fast build

release: $(TARGET.release)

debug:   $(TARGET.debug)

# ── build rules ───────────────────────────────────────────────────
$(BUILD):
	@mkdir -p $@

# pretty colourised banner
define banner
	@printf '\033[1;34m[CC]\033[0m %-26s → %s\n'
endef

$(BUILD)/$(NAME)_%: $(SRC) | $(BUILD)
	$(call banner,$<,$@)
	$(CC) $(CFLAGS.$*) $< -o $@ $(LDFLAGS)

# ── cleanup ───────────────────────────────────────────────────────
clean:
	@echo "Cleaning…"
	@rm -rf $(BUILD)
