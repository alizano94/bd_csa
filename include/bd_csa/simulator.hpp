#pragma once

#include <cstdint>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/forces.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/order_params.hpp"
#include "bd_csa/state.hpp"

namespace bd_csa {

// CPU reference implementation. Deliberately scalar and readable: its job is to
// be obviously correct so it can validate the CUDA kernels, not to be fast.
class SimulatorCpu {
 public:
  SimulatorCpu(Config cfg, MobilityTable table)
      : cfg_(std::move(cfg)), table_(std::move(table)),
        fx_(cfg_.np), fy_(cfg_.np), mobility_(cfg_.np),
        grad_d_(cfg_.np) {}

  // Advance n_steps with the field strength held at lambda.
  //
  // Overdamped Euler-Maruyama (main.f:326-328):
  //   dr = D*F*fac1*dt + sqrt(D)*xi*fac2*dt
  // which is the Ermak-McCammon scheme once fac1 = D0/kT and
  // fac2 = sqrt(2*D0/dt) in code units. The legacy drew a third Gaussian per
  // particle per step for z and discarded it (main.f:311-315); that draw is
  // simply not made here.
  void step(State& s, double lambda, long n_steps, std::uint64_t seed,
            std::uint32_t env = 0);

  [[nodiscard]] OrderParams order_params(const State& s) const {
    return compute_order_params(cfg_, s.x.data(), s.y.data(), cfg_.np);
  }

  [[nodiscard]] const Config& config() const { return cfg_; }
  void set_force_options(const ForceOptions& o) { force_opt_ = o; }

  // Per-particle D_hat as of the last refresh. Exposed so tests can predict the
  // diffusive step size from what the table actually returned rather than
  // assuming a value.
  [[nodiscard]] const std::vector<double>& mobility() const { return mobility_; }
  void refresh_mobility_now(const State& s) { refresh_mobility(s); }

 private:
  // Refresh per-particle mobility from the table, keyed on the cluster's R_g
  // and each particle's distance from the centroid (caldss.f).
  void refresh_mobility(const State& s);

  Config cfg_;
  MobilityTable table_;
  ForceOptions force_opt_{};
  std::vector<double> fx_, fy_;
  std::vector<double> mobility_;   // D_hat per particle
  std::vector<double> grad_d_;     // d(D_hat)/d(distance), for the drift term
  std::vector<double> gdx_, gdy_;  // cached gradient direction
};

}  // namespace bd_csa
