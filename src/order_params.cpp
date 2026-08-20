#include "bd_csa/order_params.hpp"

#include <cmath>
#include <vector>

namespace bd_csa {

double compute_rc(double rg_nm, double c6) {
  // conn6calc.f:143-152. Constants are tuned for np = 300 and do not transfer
  // to other particle counts.
  constexpr double kRgLB = 18526.0;
  constexpr double kRgUB = 26500.0;
  const double rg_clamped = rg_nm > kRgUB ? kRgUB : rg_nm;
  const double ra = 1.0 - (rg_clamped - kRgLB) / (kRgUB - kRgLB);
  const double crc = c6 / 5.6;
  const double wrc = 1.0 / (std::exp((crc - 0.5) * 18.0) + 1.0);
  return wrc * ra + (1.0 - wrc) * crc;
}

OrderParams compute_order_params(const Config& c, const double* x,
                                 const double* y, int np) {
  constexpr double kCtestv = 0.32;

  // Measurement window, compacted densely (the 7.2 fix).
  std::vector<double> px, py;
  px.reserve(np);
  py.reserve(np);
  for (int i = 0; i < np; ++i) {
    if (std::abs(x[i]) <= 0.5 * c.expbox[0] &&
        std::abs(y[i]) <= 0.5 * c.expbox[1]) {
      px.push_back(x[i]);
      py.push_back(y[i]);
    }
  }
  const int n = static_cast<int>(px.size());

  OrderParams op;
  op.n_in_window = n;
  if (n == 0) return op;

  // Minimum imaging is applied here only when periodicity is enabled. For the
  // shipped geometry it provably cannot change neighbour classification: the
  // largest possible separation (2 * 31570 nm) is well below dg - rmin, so no
  // pair is ever re-imaged into or out of the rmin shell.
  const double boxx = c.dg, boxy = c.dg;
  const bool wrap = c.physics.periodic;
  const auto sep = [&](double d, double box) {
    return wrap ? d - box * std::nearbyint(d / box) : d;
  };

  std::vector<double> psir(n, 0.0), psii(n, 0.0);
  std::vector<int> nb(n, 0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double dx = sep(px[j] - px[i], boxx);
      const double dy = sep(py[j] - py[i], boxy);
      if (std::sqrt(dx * dx + dy * dy) <= c.rmin) {
        ++nb[i];
        const double theta = std::atan2(dy, dx);
        psir[i] += std::cos(6.0 * theta);
        psii[i] += std::sin(6.0 * theta);
      }
    }
    if (nb[i] != 0) {
      psir[i] /= static_cast<double>(nb[i]);
      psii[i] /= static_cast<double>(nb[i]);
    }
  }

  // Phase-coherent global average. Legacy divided this by np while the loops
  // above used NPTEMP; with the dense compaction both are n, which removes the
  // inconsistency noted in 7.2.
  double accr = 0.0, acci = 0.0;
  for (int i = 0; i < n; ++i) {
    accr += psir[i];
    acci += psii[i];
  }
  accr /= n;
  acci /= n;
  op.psi6 = std::sqrt(accr * accr + acci * acci);

  // C6: count neighbours whose local hexatic phase is aligned to within
  // acos(0.32) ~ 71.3 degrees.
  double conn_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    int con = 0;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const double dx = sep(px[j] - px[i], boxx);
      const double dy = sep(py[j] - py[i], boxy);
      if (std::sqrt(dx * dx + dy * dy) > c.rmin) continue;
      const double numer = psir[i] * psir[j] + psii[i] * psii[j];
      const double cross = psii[i] * psir[j] - psii[j] * psir[i];
      const double denom = std::sqrt(numer * numer + cross * cross);
      if (denom > 0.0 && numer / denom >= kCtestv) ++con;
    }
    conn_sum += con;
  }
  op.c6 = conn_sum / n;

  // R_g about the instantaneous centroid, in double throughout (caldss.f
  // silently used REAL*4 here and so disagreed with conn6calc -- see 7.4).
  double xm = 0.0, ym = 0.0;
  for (int i = 0; i < n; ++i) {
    xm += px[i];
    ym += py[i];
  }
  xm /= n;
  ym /= n;

  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    var += (px[i] - xm) * (px[i] - xm) + (py[i] - ym) * (py[i] - ym);
  }
  op.rg = std::sqrt(var / n);
  op.rc = compute_rc(op.rg, op.c6);
  return op;
}

}  // namespace bd_csa
