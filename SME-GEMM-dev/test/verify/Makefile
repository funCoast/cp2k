CXX = clang++
HOST_ARCH_FLAGS ?= -march=armv8-a
SME_ARCH_FLAGS ?= -march=armv8-a+sme+sme2+sve2+sme-f64f64
COMMON_FLAGS ?= -O3 -g
CPPFLAGS += -I../../include
KERNEL_SRC := $(firstword $(wildcard kernel.s kernel.cpp))

verify: verify.o kernel.o state.o wrapper.o
	$(CXX) -o verify verify.o kernel.o state.o wrapper.o $(COMMON_FLAGS)

verify.o: verify.cpp shape.h
	$(CXX) -c verify.cpp $(CPPFLAGS) -std=c++20 $(HOST_ARCH_FLAGS) $(COMMON_FLAGS)

kernel.o: $(KERNEL_SRC)
	$(CXX) -c $(KERNEL_SRC) $(CPPFLAGS) -std=c++20 $(SME_ARCH_FLAGS) $(COMMON_FLAGS)

wrapper.o: wrapper.cpp shape.h
	$(CXX) -c wrapper.cpp $(CPPFLAGS) -std=c++20 $(SME_ARCH_FLAGS) $(COMMON_FLAGS)

state.o: ../state.S
	$(CXX) -c ../state.S -o state.o $(CPPFLAGS) -std=c++20 $(SME_ARCH_FLAGS) $(COMMON_FLAGS)

clean:
	rm -f *.o verify
