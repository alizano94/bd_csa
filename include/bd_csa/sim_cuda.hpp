#pragma once

#include <cstdint>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/state.hpp"

namespace bd_csa {

// CUDA backend. One block per environment, the whole episode in one kernel
// launch, positions resident in shared memory throughout.
//
// Precision follows the split established in the plan: FP64 for positions and
// the displacement accumulator, FP32 for the O(N^2) force loop. On consumer Ada
// FP64 runs at 1/64 rate, so keeping the force loop in FP32 is what makes the
// port worthwhile; keeping the accumulator in FP64 is what stops 10^6 tiny
// increments from washing out against coordinates of order 2e4 nm.
class SimulatorCuda {
 public:
  SimulatorCuda(const Config& cfg, const MobilityTable& table, int n_envs = 1);
  ~SimulatorCuda();
  SimulatorCuda(const SimulatorCuda&) = delete;
  SimulatorCuda& operator=(const SimulatorCuda&) = delete;

  static bool available();

  void upload(const State& s, int env = 0);
  void download(State& s, int env = 0) const;

  void step(double lambda, long n_steps, std::uint64_t seed);

  // Single force evaluation, interleaved (fx, fy) per particle per env.
  // Used by the tier-4 test to compare against the FP64 CPU reference.
  std::vector<float> forces_once(double lambda, bool legacy_overlap = false);

  [[nodiscard]] size_t shared_bytes() const;

 private:
  struct Impl;
  Impl* impl_;
  Config cfg_;
};

}  // namespace bd_csa
