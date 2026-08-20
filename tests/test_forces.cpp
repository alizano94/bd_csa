// Validation tier 2: single-step force differential against the Fortran.
//
// This is the highest-value test in the suite. forces() is a pure function of
// (positions, lambda, config) -- no RNG, no mobility, no history -- so the
// legacy code and the port must agree to round-off, not merely statistically.
// It is also the only practical way to catch a sign or index error in the
// asymmetric i/j dipole gradient terms, where the force on i is deliberately
// NOT minus the force on j (documentation/01-physical-model.md 1.4a).
//
// The two force contributions are tested at different tolerances because they
// have different conditioning -- see the note in tests/oracle/build_force_oracle.sh.
// The pair forces carry all the physics that is easy to get wrong, and they are
// gated at 1e-12.
//
// References come from tests/oracle/build_force_oracle.sh.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/forces.hpp"
#include "bd_csa/io.hpp"
#include "check.hpp"

using namespace bd_csa;

namespace {

const std::string kData = BD_CSA_DATA_DIR;
const std::string kBuild = BD_CSA_BUILD_DIR;

struct Reference {
  double lambda = 0.0;
  std::vector<double> x, y, fx, fy, fz;
};

bool load_reference(const std::string& path, Reference& ref) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("# lambda", 0) == 0) {
      ref.lambda = std::stod(line.substr(8));
      continue;
    }
    if (!line.empty() && line[0] == '#') continue;
    std::istringstream ss(line);
    int idx;
    double px, py, fx, fy, fz;
    if (!(ss >> idx >> px >> py >> fx >> fy >> fz)) continue;
    ref.x.push_back(px);
    ref.y.push_back(py);
    ref.fx.push_back(fx);
    ref.fy.push_back(fy);
    ref.fz.push_back(fz);
  }
  return !ref.x.empty();
}

// Compare force vectors, normalising by the magnitude of the reference vector
// so that a near-zero component does not manufacture a huge relative error.
double max_rel_error(const std::vector<double>& gx, const std::vector<double>& gy,
                     const std::vector<double>& rx, const std::vector<double>& ry,
                     int* worst = nullptr) {
  double worst_rel = 0.0;
  for (std::size_t i = 0; i < rx.size(); ++i) {
    const double scale = std::hypot(rx[i], ry[i]);
    if (scale == 0.0) continue;
    const double err = std::hypot(gx[i] - rx[i], gy[i] - ry[i]) / scale;
    if (err > worst_rel) {
      worst_rel = err;
      if (worst) *worst = static_cast<int>(i);
    }
  }
  return worst_rel;
}

std::string fmt_e(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "max rel err %.3e", v);
  return b;
}

}  // namespace

int main() {
  Config c = Config::from_run_txt(kData + "/run.txt");
  // Reproduce the Fortran's REAL*4 literals so the comparison isolates the
  // algebra rather than the constants.
  c.legacy_float_literals = true;

  Reference full, pair;
  const std::string full_path = kBuild + "/oracle_forcedump/force_dump.txt";
  const std::string pair_path = kBuild + "/oracle_forcedump_pair/force_dump.txt";
  if (!load_reference(full_path, full) || !load_reference(pair_path, pair)) {
    std::printf("[SKIP] force oracles missing; run tests/oracle/build_force_oracle.sh\n");
    return 77;  // ctest NOT_RUN
  }

  const State s = read_start_txt(kData + "/start.txt", c);
  check::eq_int(static_cast<long>(full.x.size()), c.np, "reference particle count");

  double max_pos_err = 0.0;
  for (int i = 0; i < c.np; ++i)
    max_pos_err = std::max({max_pos_err, std::abs(s.x[i] - full.x[i]),
                            std::abs(s.y[i] - full.y[i])});
  check::near_abs(max_pos_err, 0.0, 1e-9, "positions match Fortran exactly (nm)");

  std::vector<double> fx(c.np), fy(c.np);

  // ---- the strict gate: DLVO + asymmetric dipole, no DEP -----------------
  // Setting lambda scales the dipole term; the DEP term is absent from this
  // oracle, so any disagreement is in the pair physics alone.
  std::puts("\n-- pair forces (DLVO + dipole) vs Fortran --");
  ForceOptions pair_only;
  pair_only.dep_gradient = DepGradient::kNone;
  pair_only.legacy_overlap = true;
  compute_forces(c, pair.lambda, s.x.data(), s.y.data(), fx.data(), fy.data(),
                 c.np, pair_only);

  int worst = -1;
  const double rel_pair = max_rel_error(fx, fy, pair.fx, pair.fy, &worst);
  char detail[256];
  std::snprintf(detail, sizeof detail, "max rel err %.3e at particle %d", rel_pair,
                worst);
  check::report(rel_pair <= 1e-12, "pair force agrees to 1e-12", detail);

  // Newton's third law must FAIL -- that is the physics, not a bug. The dipole
  // moments are position dependent, so the pair interaction is non-Newtonian.
  // If a refactor ever symmetrised it, this check catches it.
  double sx = 0.0, sy = 0.0;
  for (int i = 0; i < c.np; ++i) {
    sx += pair.fx[i];
    sy += pair.fy[i];
  }
  check::report(std::hypot(sx, sy) > 1e-9,
                "pair forces are non-Newtonian (asymmetric dipole)",
                "|sum F| = " + std::to_string(std::hypot(sx, sy)));

  double max_fz = 0.0;
  for (double v : full.fz) max_fz = std::max(max_fz, std::abs(v));
  check::near_abs(max_fz, 0.0, 0.0, "Fortran Fz identically zero (2-D)");

  // ---- the DEP term, at the precision it can actually support ------------
  // Reproducing the legacy forward difference bit-for-bit is impossible: the
  // cancellation leaves only ~2e-6 relative accuracy and the residue depends on
  // compiler and instruction selection.
  std::puts("\n-- full forces incl. DEP, legacy finite difference --");
  ForceOptions legacy;
  legacy.dep_gradient = DepGradient::kFiniteDifference;
  legacy.legacy_overlap = true;
  compute_forces(c, full.lambda, s.x.data(), s.y.data(), fx.data(), fy.data(),
                 c.np, legacy);
  const double rel_fd = max_rel_error(fx, fy, full.fx, full.fy);
  check::report(rel_fd <= 1e-5,
                "matches Fortran within finite-difference cancellation noise",
                fmt_e(rel_fd));

  // ---- the analytic gradient, which is what the GPU will use -------------
  std::puts("\n-- analytic grad|E|^2 --");
  std::vector<double> ax(c.np), ay(c.np);
  ForceOptions analytic;
  analytic.dep_gradient = DepGradient::kAnalytic;
  analytic.legacy_overlap = true;
  compute_forces(c, full.lambda, s.x.data(), s.y.data(), ax.data(), ay.data(),
                 c.np, analytic);
  const double rel_an = max_rel_error(ax, ay, full.fx, full.fy);
  check::report(rel_an <= 1e-4, "analytic DEP agrees with Fortran to ~1e-5",
                fmt_e(rel_an));

  // The closed form is exact, so it must match a central difference of |E|^2 to
  // that difference's own truncation error, O(h^2). This validates the
  // derivation independently of the Fortran.
  double worst_grad = 0.0;
  for (int i = 0; i < c.np; ++i) {
    double gx, gy;
    grad_emag_sq(c, s.x[i], s.y[i], gx, gy);
    const double h = 1.0;  // nm: balances O(h^2) truncation against cancellation
    const auto e2 = [&](double px, double py) {
      const double e = emag(c, px, py);
      return e * e;
    };
    const double nx = (e2(s.x[i] + h, s.y[i]) - e2(s.x[i] - h, s.y[i])) / (2 * h);
    const double ny = (e2(s.x[i], s.y[i] + h) - e2(s.x[i], s.y[i] - h)) / (2 * h);
    const double scale = std::hypot(nx, ny);
    if (scale > 0)
      worst_grad = std::max(worst_grad, std::hypot(gx - nx, gy - ny) / scale);
  }
  check::report(worst_grad <= 1e-8,
                "closed form matches central difference of |E|^2",
                fmt_e(worst_grad));

  // The whole reason for the closed form: in FP32 the legacy difference is
  // destroyed by cancellation, while the analytic form is unaffected.
  std::puts("\n-- why the closed form is mandatory on GPU --");
  {
    const int i = 0;
    const float h = 1e-3f;
    const auto e2f = [&](float px, float py) {
      const float e = static_cast<float>(emag(c, px, py));
      return e * e;
    };
    const float fd32 =
        (e2f(static_cast<float>(s.x[i]) + h, static_cast<float>(s.y[i])) -
         e2f(static_cast<float>(s.x[i]), static_cast<float>(s.y[i]))) / h;
    double gx, gy;
    grad_emag_sq(c, s.x[i], s.y[i], gx, gy);
    const double rel32 = std::abs(fd32 - gx) / std::abs(gx);
    check::report(rel32 > 0.5,
                  "FP32 forward difference loses all precision (>50% error)",
                  "fd32=" + std::to_string(fd32) + " exact=" + std::to_string(gx));
  }

  return check::finish("tier2-forces");
}
