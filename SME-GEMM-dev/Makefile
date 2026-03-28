BUILD_DIR ?= out/build/default
CONFIGURE_PRESET ?= default
CMAKE ?= $(shell command -v cmake 2>/dev/null || echo cmake)
HPCKIT_ROOT ?=

.PHONY: configure build clean thirdparty rebuild-thirdparty

configure:
	$(CMAKE) --preset $(CONFIGURE_PRESET)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean

thirdparty:
	bash scripts/build_thirdparty.sh build

rebuild-thirdparty:
	bash scripts/build_thirdparty.sh rebuild

freq_test: freq_test.cpp
	@if [ -z "$(HPCKIT_ROOT)" ]; then \
		echo "Set HPCKIT_ROOT before building freq_test."; \
		exit 1; \
	fi
	g++ freq_test.cpp -O3 -o freq_test -march=armv8-a+sme-f64f64+sme+sve2 -I$(HPCKIT_ROOT)/include -L$(HPCKIT_ROOT)/lib -g
