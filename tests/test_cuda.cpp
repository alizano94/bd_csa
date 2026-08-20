// Validation tier 4: the CUDA backend against the FP64 CPU reference.
//
// The CPU path is already pinned to the Fortran at 1e-14 (tier 2), so it is a
// trustworthy reference here. What this test establishes is the FP32 error
// budget of the force kernel and that the integrator agrees statistically.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/forces.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/sim_cuda.hpp"
#include "bd_csa/simulator.hpp"
#include "check.hpp"

using namespace bd_csa;

namespace {
const std::string kData = BD_CSA_DATA_DIR;

std::string fmt(const char* label, double v) {
  char b[96];
  std::snprintf(b, sizeof b, "%s %.3e", label, v);
  return b;
}
}  // namespace

int main() {
  if (!SimulatorCuda::available()) {
    std::printf("[SKIP] no CUDA device\n");
    return 77;
  }

  const Config c = Config::from_run_txt(kData + "/run.txt");
  const MobilityTable table = MobilityTable::load(kData + "/2dtabledssnp300.txt", c);
  const State s0 = read_start_txt(kData + "/start.txt", c);
  const double lambda = 30.0;

  SimulatorCuda gpu(c, table, /*n_envs=*/1);
  gpu.upload(s0);
  std::printf("shared memory per block: %zu bytes\n", gpu.shared_bytes());

  // ---- force kernel vs FP64 CPU -----------------------------------------
  std::puts("\n-- force kernel: FP32 GPU vs FP64 CPU --");
  std::vector<double> cx(c.np), cy(c.np);
  ForceOptions opt;
  opt.dep_gradient = DepGradient::kAnalytic;  // what the GPU uses
  compute_forces(c, lambda, s0.x.data(), s0.y.data(), cx.data(), cy.data(), c.np,
                 opt);

  const std::vector<float> g = gpu.forces_once(lambda);

  // Normalise by the RMS force over all particles, not by each particle's own
  // magnitude. Some particles sit near a force null where the inward DEP trap
  // cancels the pair repulsion -- particle 175 carries ~1/99 of the mean force
  // -- and dividing by that manufactures a huge relative error from an
  // absolute one that is irrelevant to the dynamics. What moves a particle is
  // the error compared to a typical force.
  double fscale2 = 0.0;
  for (int i = 0; i < c.np; ++i) fscale2 += cx[i] * cx[i] + cy[i] * cy[i];
  const double fscale = std::sqrt(fscale2 / c.np);

  double worst_abs = 0.0, sum2 = 0.0, worst_self = 0.0;
  int worst_i = -1;
  for (int i = 0; i < c.np; ++i) {
    const double e = std::hypot(g[2 * i] - cx[i], g[2 * i + 1] - cy[i]);
    sum2 += e * e;
    if (e > worst_abs) { worst_abs = e; worst_i = i; }
    const double self = std::hypot(cx[i], cy[i]);
    if (self > 0) worst_self = std::max(worst_self, e / self);
  }
  const double rms = std::sqrt(sum2 / c.np) / fscale;
  const double worst = worst_abs / fscale;
  std::printf("       RMS force scale %.3e, worst at particle %d\n", fscale, worst_i);
  std::printf("       (per-particle-normalised max would be %.3e, inflated by\n"
              "        near-null particles -- see comment)\n", worst_self);

  // FP32 carries ~1.2e-7 relative resolution and the pair sum partially
  // cancels, so ~1e-5 against the typical force scale is the expected budget.
  // Anything beyond 1e-4 would indicate an algorithmic difference, not rounding.
  check::report(worst <= 1e-4, "max force error vs typical force scale",
                fmt("max", worst));
  check::report(rms <= 2e-5, "RMS force error vs typical force scale",
                fmt("rms", rms));

  // ---- integrator: statistical agreement ---------------------------------
  // Trajectories cannot match pointwise (FP32 forces diverge chaotically), so
  // compare the observables that actually define the RL contract.
  std::puts("\n-- 2000 steps: observables vs CPU --");
  SimulatorCpu cpu(c, table);
  State scpu = s0;
  cpu.step(scpu, lambda, 2000, 7);
  const OrderParams ocpu = cpu.order_params(scpu);

  gpu.upload(s0);
  gpu.step(lambda, 2000, 7);
  State sgpu(c.np);
  gpu.download(sgpu);
  const OrderParams ogpu = compute_order_params(c, sgpu.x.data(), sgpu.y.data(), c.np);

  std::printf("       CPU: R_g %.1f  psi6 %.4f  C6 %.3f\n", ocpu.rg, ocpu.psi6, ocpu.c6);
  std::printf("       GPU: R_g %.1f  psi6 %.4f  C6 %.3f\n", ogpu.rg, ogpu.psi6, ogpu.c6);

  check::report(std::abs(ogpu.rg - ocpu.rg) / ocpu.rg < 0.02,
                "R_g agrees within 2%", fmt("rel", std::abs(ogpu.rg - ocpu.rg) / ocpu.rg));
  check::report(std::abs(ogpu.c6 - ocpu.c6) < 0.5, "C6 agrees within 0.5",
                fmt("abs", std::abs(ogpu.c6 - ocpu.c6)));
  check::report(std::abs(ogpu.psi6 - ocpu.psi6) < 0.1, "psi6 agrees within 0.1",
                fmt("abs", std::abs(ogpu.psi6 - ocpu.psi6)));

  // Physical sanity, same checks the CPU integrator has to pass.
  double min_sep = 1e30;
  for (int i = 0; i < c.np; ++i)
    for (int j = i + 1; j < c.np; ++j)
      min_sep = std::min(min_sep, std::hypot(sgpu.x[i] - sgpu.x[j],
                                             sgpu.y[i] - sgpu.y[j]));
  check::report(min_sep > 2.0 * c.a, "no pair reaches contact",
                fmt("min sep (a)", min_sep / c.a));
  check::report(ogpu.rg < 0.5 * c.dg, "cluster stays trapped", fmt("R_g", ogpu.rg));

  // ---- determinism --------------------------------------------------------
  std::puts("\n-- determinism --");
  gpu.upload(s0);
  gpu.step(lambda, 200, 99);
  State a(c.np);
  gpu.download(a);
  gpu.upload(s0);
  gpu.step(lambda, 200, 99);
  State b(c.np);
  gpu.download(b);
  double d = 0.0;
  for (int i = 0; i < c.np; ++i) d = std::max(d, std::abs(a.x[i] - b.x[i]));
  check::near_abs(d, 0.0, 0.0, "same seed -> bit-identical on GPU");

  return check::finish("tier4-cuda");
}
