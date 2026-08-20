#pragma once

#include <vector>

namespace bd_csa {

// Per-environment mutable state. Structure-of-arrays because the CUDA backend
// wants coalesced access and the z coordinate is frozen (the system is 2-D:
// every particle sits at z = hlev, so z carries no information and no force).
// This is the whole of the simulation state -- a step is a pure function of
// (positions, seed, lambda), which is what makes batching possible.
struct State {
  std::vector<double> x;  // nm
  std::vector<double> y;  // nm

  explicit State(int np) : x(np, 0.0), y(np, 0.0) {}

  [[nodiscard]] int size() const { return static_cast<int>(x.size()); }
};

}  // namespace bd_csa
