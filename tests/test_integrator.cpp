// Integrator and RNG checks: the statistical properties the Euler-Maruyama step
// must have, plus the defects in the legacy RNG that motivated replacing it.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/rng.hpp"
#include "bd_csa/simulator.hpp"
#include "check.hpp"

using namespace bd_csa;

namespace {

const std::string kData = BD_CSA_DATA_DIR;

void test_rng() {
  std::puts("\n-- counter-based RNG --");
  const int n = 200000;
  double sum = 0, sum2 = 0, sum3 = 0, sum4 = 0;
  for (int i = 0; i < n / 2; ++i) {
    double a, b;
    Philox::normal2(12345, 0, 0, static_cast<std::uint64_t>(i), a, b);
    for (double v : {a, b}) {
      sum += v;
      sum2 += v * v;
      sum3 += v * v * v;
      sum4 += v * v * v * v;
    }
  }
  const double mean = sum / n;
  const double var = sum2 / n - mean * mean;
  const double skew = sum3 / n;
  const double kurt = sum4 / n;
  check::near_abs(mean, 0.0, 0.01, "mean ~ 0");
  check::near_abs(var, 1.0, 0.01, "variance ~ 1");
  check::near_abs(skew, 0.0, 0.02, "skewness ~ 0");
  check::near_abs(kurt, 3.0, 0.05, "kurtosis ~ 3");

  // Reproducibility: same key -> same value, independent of call order.
  double a1, b1, a2, b2;
  Philox::normal2(7, 0, 42, 999, a1, b1);
  Philox::normal2(7, 0, 13, 5, a2, b2);   // unrelated draw in between
  double a3, b3;
  Philox::normal2(7, 0, 42, 999, a3, b3);
  check::near_abs(a3, a1, 0.0, "same key reproduces exactly (order independent)");
  check::near_abs(b3, b1, 0.0, "  ... second component too");

  // The legacy defect: distinct positive seeds produced identical opening
  // deviates. Distinct seeds must now diverge from the very first draw.
  double s1a, s1b, s2a, s2b;
  Philox::normal2(1, 0, 0, 0, s1a, s1b);
  Philox::normal2(7, 0, 0, 0, s2a, s2b);
  check::report(std::abs(s1a - s2a) > 1e-6,
                "seeds 1 and 7 differ at the FIRST draw (7.1 fixed)",
                "got " + std::to_string(s1a) + " vs " + std::to_string(s2a));

  // Distinct particles in the same step must be independent too.
  double p0a, p0b, p1a, p1b;
  Philox::normal2(7, 0, 0, 0, p0a, p0b);
  Philox::normal2(7, 0, 1, 0, p1a, p1b);
  check::report(std::abs(p0a - p1a) > 1e-6, "per-particle streams are distinct",
                "got " + std::to_string(p0a) + " vs " + std::to_string(p1a));
}

// With forces switched off the scheme reduces to pure Brownian motion, whose
// mean-square displacement after n steps must be n * (sqrt(D)*fac2*dt)^2 per
// axis. This is the cleanest possible check that the noise amplitude, and hence
// the fac2 calibration, is wired up correctly.
void test_free_diffusion(const Config& base, const MobilityTable& table) {
  std::puts("\n-- free diffusion (forces off) --");
  Config c = base;
  c.physics.enable_divD_drift = false;
  c.physics.mobility_update_interval = 0;  // freeze mobility
  c.fcm = -0.4667;

  SimulatorCpu sim(c, table);
  ForceOptions off;
  off.dep_gradient = DepGradient::kNone;
  sim.set_force_options(off);

  // Zero lambda kills the dipole force; the DLVO term is still present, so
  // start from a dilute lattice where particles are far beyond rcut = 5a.
  const int np = c.np;
  State s(np);
  const double spacing = 20.0 * c.a;  // >> rcut
  for (int i = 0; i < np; ++i) {
    s.x[i] = (i % 20) * spacing;
    s.y[i] = (i / 20) * spacing;
  }
  const State s0 = s;
  sim.refresh_mobility_now(s0);

  // Average over several independent seeds. With np particles x 2 axes x
  // n_seeds samples the relative standard error of the MSD estimate is
  // sqrt(2/(2*np*n_seeds)); at 10 seeds that is ~1.8%, so a 5% band is a ~2.7
  // sigma test rather than a coin flip.
  const long steps = 200;
  const int n_seeds = 10;
  double msd = 0.0;
  for (int k = 0; k < n_seeds; ++k) {
    State sk = s0;
    sim.step(sk, /*lambda=*/0.0, steps, /*seed=*/2024 + k);
    for (int i = 0; i < np; ++i) {
      const double dx = sk.x[i] - s0.x[i], dy = sk.y[i] - s0.y[i];
      msd += dx * dx + dy * dy;
    }
  }
  msd /= (2.0 * np * n_seeds);  // per axis

  // Predicted: n * <D_hat> * (fac2*dt)^2. Mobility is frozen at the initial
  // configuration and varies per particle (they sit in different distance
  // bins), so average what the table actually returned rather than assuming a
  // single value.
  double dhat_mean = 0.0;
  for (double d : sim.mobility()) dhat_mean += d;
  dhat_mean /= np;
  const double expected = steps * dhat_mean * (c.fac2 * c.dt) * (c.fac2 * c.dt);

  char buf[160];
  std::snprintf(buf, sizeof buf, "msd=%.4g expected=%.4g ratio=%.4f", msd, expected,
                msd / expected);
  // ~5% band: 300 particles x 400 steps gives a relative s.e. of ~1/sqrt(2*N*n).
  check::report(std::abs(msd / expected - 1.0) < 0.05,
                "MSD matches n*D*(fac2*dt)^2 within sampling error", buf);
}

// A run must be reproducible from its seed, and different seeds must produce
// different trajectories.
void test_determinism(const Config& base, const MobilityTable& table) {
  std::puts("\n-- determinism --");
  SimulatorCpu sim(base, table);
  const State s0 = read_start_txt(kData + "/start.txt", base);

  State a = s0, b = s0, cc = s0;
  sim.step(a, 30.0, 50, 111);
  sim.step(b, 30.0, 50, 111);
  sim.step(cc, 30.0, 50, 222);

  double same = 0.0, diff = 0.0;
  for (int i = 0; i < base.np; ++i) {
    same = std::max(same, std::abs(a.x[i] - b.x[i]));
    diff = std::max(diff, std::abs(a.x[i] - cc.x[i]));
  }
  check::near_abs(same, 0.0, 0.0, "same seed -> bit-identical trajectory");
  check::report(diff > 1e-9, "different seed -> different trajectory",
                "max |dx| = " + std::to_string(diff));
}

// A short run from the shipped configuration must stay physically sane: the
// cluster stays trapped and no pair collapses to overlap.
void test_stability(const Config& base, const MobilityTable& table) {
  std::puts("\n-- short run stays physical --");
  SimulatorCpu sim(base, table);
  State s = read_start_txt(kData + "/start.txt", base);
  const OrderParams before = sim.order_params(s);

  sim.step(s, 30.0, 2000, 7);
  const OrderParams after = sim.order_params(s);

  check::report(std::isfinite(after.rg) && after.rg > 0, "R_g finite and positive",
                "R_g = " + std::to_string(after.rg));
  check::report(after.rg < 0.5 * base.dg, "cluster stays trapped inside the cell",
                "R_g = " + std::to_string(after.rg) + " nm");
  check::report(after.psi6 >= 0.0 && after.psi6 <= 1.0, "psi6 in [0,1]",
                "psi6 = " + std::to_string(after.psi6));
  check::report(after.c6 >= 0.0 && after.c6 <= 6.0, "C6 in [0,6]",
                "C6 = " + std::to_string(after.c6));

  double min_sep = 1e30;
  for (int i = 0; i < base.np; ++i)
    for (int j = i + 1; j < base.np; ++j)
      min_sep = std::min(min_sep, std::hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]));
  // The 10 nm Debye screening should stop particles well before contact at 2a;
  // the docs report 2.052a as the minimum observed over a full run.
  check::report(min_sep > 2.0 * base.a,
                "no pair reaches contact (overlap branch never fires)",
                "min separation = " + std::to_string(min_sep / base.a) + " a");

  std::printf("       R_g %.1f -> %.1f nm, psi6 %.4f -> %.4f, C6 %.3f -> %.3f\n",
              before.rg, after.rg, before.psi6, after.psi6, before.c6, after.c6);
}

}  // namespace

int main() {
  const Config c = Config::from_run_txt(kData + "/run.txt");
  const MobilityTable table = MobilityTable::load(kData + "/2dtabledssnp300.txt", c);

  test_rng();
  test_free_diffusion(c, table);
  test_determinism(c, table);
  test_stability(c, table);
  return check::finish("integrator");
}
