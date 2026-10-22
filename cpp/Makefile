# -ffp-contract=off: no fused multiply-adds, so scores round exactly as numpy / CPython round them
CXXFLAGS ?= -O3 -march=native -std=c++20 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter

replay_srbh: replay_srbh.cpp sse.hpp
	$(CXX) $(CXXFLAGS) -o $@ replay_srbh.cpp -lquadmath

clean:
	rm -f replay_srbh
