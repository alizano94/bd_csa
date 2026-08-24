// Validation tier 1: config parsing, calibration-constant derivation, mobility
// table loading, and the t=0 order parameters.
//
// Golden values are the ones the legacy binary actually prints for the shipped
// data/start.txt, recorded in data/GOLDEN.md. They are NOT the values quoted in
// documentation/ -- those refer to a different configuration; see the
// correction note in data/GOLDEN.md.

#include <cmath>
#include <string>

#include "bd_csa/config.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/order_params.hpp"
#include "check.hpp"

using namespace bd_csa;

namespace {

const std::string kData = BD_CSA_DATA_DIR;

void test_run_txt_parse(const Config& c) {
  std::puts("\n-- run.txt parse (positional, mirrors main.f:74-156) --");
  check::eq_int(c.np, 300, "np");
  check::eq_int(c.nstep, 1000000, "nstep");
  check::eq_int(c.iprint, 1000000, "iprint");
  check::eq_str(c.par_in, "start.txt", "par_in");
  check::eq_str(c.par_out, "bd_xyz", "par_out");
  check::near_abs(c.a, 1435.0, 1e-12, "a (nm)");
  check::near_abs(c.tempr, 20.0, 1e-12, "tempr (C)");
  check::near_abs(c.dt, 0.1, 1e-12, "dt (ms)");
  check::eq_int(c.check, 'n', "check flag");
  check::near_abs(c.kappa, 143.5, 1e-12, "kappa*a");
  check::near_abs(c.fcm, -0.4667, 1e-12, "Clausius-Mossotti");
  check::near_abs(c.rmin, 3780.0, 1e-12, "rmin (nm)");
  check::near_abs(c.var, 0.0, 1e-12, "measurement noise var");
  check::eq_str(c.polymono, "mono", "polymono");
  check::eq_str(c.rgdsfile, "2dtabledssnp300.txt", "rgdsfile");
  check::eq_int(c.ecorrectflag, 1, "ecorrectflag");
  check::near_abs(c.dpf, 1.0, 1e-12, "dpf");

  // Rescalings applied on load (main.f:170-177).
  check::near_abs(c.rcut, 5.0 * 1435.0, 1e-9, "rcut = 5a (nm)");
  check::near_abs(c.re, 5.0 * 1435.0, 1e-9, "re = 5a (nm)");
  check::near_abs(c.dg, 63.415 * 1435.0, 1e-9, "dg = 63.415a (nm)");
  check::near_abs(c.hlev, 1435.0 * 1.0663, 1e-9, "hlev = a(1+h) (nm)");
  check::near_abs(c.pfpp, 2.2975 * 1435.0, 1e-9, "pfpp*a");
  check::near_abs(c.expbox[0], 63.415 * 1435.0, 1e-9, "expbox x (nm)");
  check::near_abs(c.expbox[1], 63.415 * 1435.0, 1e-9, "expbox y (nm)");

  // The mobility-table binning: note the NEGATIVE R_g stride.
  check::near_abs(c.rgdsmin, 26500.0, 1e-12, "rgdsmin (nm)");
  check::near_abs(c.delrgdsmin, -250.0, 1e-12, "delrgdsmin (nm, negative)");
  check::eq_int(c.rgdssbin, 30, "rgdssbin");
  check::near_abs(c.deldist, 1435.0, 1e-12, "deldist (nm)");
  check::eq_int(c.distdssbin, 50, "distdssbin");
  check::near_abs(c.dssmin, 0.10, 1e-12, "dssmin");
  check::near_abs(c.dssmax, 0.40, 1e-12, "dssmax");
}

// documentation/02-numerical-methods.md 2.3: fac1 and fac2 are undocumented
// magic numbers in run.txt. They are actually D0/kT and sqrt(2*D0/dt) in code
// units, which is what makes the port's unit system checkable rather than
// copied on faith.
void test_calibration(const Config& c) {
  std::puts("\n-- fac1/fac2 vs Stokes-Einstein --");

  check::near_abs(c.fac1_raw, 5.9582e7, 1e2, "fac1 raw (run.txt)");
  check::near_abs(c.fac2_raw, 40.5622, 1e-9, "fac2 raw (run.txt)");
  check::near_rel(c.fac1, 5.9582e7 / 1435.0, 1e-12, "fac1 = raw/a");

  const double D0 = c.D0_nm2_per_s();  // nm^2/s at eta = 0.890 mPa.s
  check::near_rel(D0, 1.6803e5, 1e-3, "D0 (nm^2/s)");

  // Deterministic displacement produced by one reduced force unit in one step.
  const double code_det = c.fac1 * c.dt * c.force_scale();
  const double theory_det = D0 * (c.dt * 1e-3) / c.a;  // dt ms -> s
  check::near_rel(code_det, theory_det, 1e-3,
                  "deterministic step: fac1 vs D0*dt/a");
  check::near_abs(theory_det, 0.011709, 5e-6, "theory det. step = 0.011709 nm");

  // RMS random displacement per step, per sqrt(D_hat).
  const double code_rand = c.fac2 * c.dt;
  const double theory_rand = std::sqrt(2.0 * D0 * (c.dt * 1e-3));
  check::near_rel(code_rand, theory_rand, 1e-3,
                  "random step: fac2 vs sqrt(2*D0*dt)");
  check::near_abs(theory_rand, 5.797, 1e-3, "theory rand. step = 5.797 nm");

  // Reduced-force scale 1e18*kB*T/a (documentation 02 section 2.3).
  check::near_rel(c.force_scale(), 2.81905e-6, 1e-4, "force scale 1e18*kT/a");
}

void test_mobility_table(const Config& c) {
  std::puts("\n-- mobility table --");
  const auto t = MobilityTable::load(kData + "/2dtabledssnp300.txt", c);
  check::eq_int(t.rows(), 30, "rows (R_g bins)");
  check::eq_int(t.cols(), 50, "cols (distance bins)");

  // First and last records of the file, verified by inspection.
  check::near_abs(t.at(0, 0), 0.28580, 1e-9, "D_hat[0][0]");
  check::eq_int(t.count_at(0, 0), 4, "count[0][0]");
  check::near_abs(t.at(0, 1), 0.28582, 1e-9, "D_hat[0][1]");

  // caldss.f binning for the golden configuration: R_g = 21015.90986 gives
  // int((21015.90986 - 26500)/(-250)) + 1 = int(21.936) + 1 = 22 (1-based).
  const double rg = 21015.90986;
  const int rg_bin_1based = static_cast<int>((rg - c.rgdsmin) / c.delrgdsmin) + 1;
  check::eq_int(rg_bin_1based, 22, "R_g bin index (1-based) for golden config");

  // A particle at the centroid falls in distance bin 1; D_hat must be a real
  // tabulated value, not a fallback.
  const double d_centre = t.lookup_nearest(rg, 0.0);
  check::near_abs(d_centre, t.at(21, 0), 1e-12, "lookup at centroid == table");
  check::report(d_centre >= c.dssmin && d_centre <= c.dssmax,
                "D_hat within [dssmin, dssmax]", "value " + std::to_string(d_centre));

  // Out-of-range rules (caldss.f:49-68).
  check::near_abs(t.lookup_nearest(rg, 1e9), c.dssmax, 1e-12,
                  "distance beyond table -> dssmax");
  check::near_abs(t.lookup_nearest(1000.0, 0.0), c.dssmin, 1e-12,
                  "R_g more compact than table -> dssmin");

  // The smooth variant agrees with the nearest-bin lookup only where BOTH axes
  // sit at a bin centre, since that is where the interpolation weights collapse.
  // Note the golden R_g (21015.9) is *not* at a centre -- it lies 43.6% into
  // bin 22 -- so the centre must be constructed explicitly on each axis.
  const double rg_centre = c.rgdsmin + 21.5 * c.delrgdsmin;  // centre of bin 22
  const double dist_centre = c.distmin + 0.5 * c.deldist;
  double grad = 0.0;
  const double smooth = t.lookup_smooth(rg_centre, dist_centre, &grad);
  check::near_abs(smooth, t.lookup_nearest(rg_centre, dist_centre), 1e-9,
                  "smooth == nearest at bin centre (both axes)");
  check::report(std::isfinite(grad), "smooth gradient is finite",
                "d(D_hat)/d(dist) = " + std::to_string(grad));

  // Off-centre, the two must differ -- otherwise the interpolation is not
  // actually smoothing anything and the div.D term would be identically zero.
  const double off = c.distmin + 1.3 * c.deldist;
  check::report(std::abs(t.lookup_smooth(rg, off) - t.lookup_nearest(rg, off)) > 1e-9,
                "smooth differs from nearest off-centre",
                "smooth=" + std::to_string(t.lookup_smooth(rg, off)) +
                    " nearest=" + std::to_string(t.lookup_nearest(rg, off)));

  // Central-difference check of the analytic gradient away from a centre.
  const double probe = off;
  const double h = 1.0;  // nm
  double g_analytic = 0.0;
  (void)t.lookup_smooth(rg, probe, &g_analytic);
  const double g_numeric =
      (t.lookup_smooth(rg, probe + h) - t.lookup_smooth(rg, probe - h)) / (2 * h);
  check::near_abs(g_analytic, g_numeric, 1e-9,
                  "analytic vs central-difference d(D_hat)/d(dist)");
}

// Golden values from data/GOLDEN.md, produced by the legacy binary and proven
// invariant to lambda and seed.
void test_order_params(const Config& c) {
  std::puts("\n-- t=0 order parameters vs oracle --");
  const State s = read_start_txt(kData + "/start.txt", c);
  check::eq_int(s.size(), 300, "particles read from start.txt");

  const OrderParams op = compute_order_params(c, s.x.data(), s.y.data(), c.np);
  check::eq_int(op.n_in_window, 300, "all particles inside expbox");

  // op1.txt prints five decimals, so match to half an ulp of that.
  check::near_abs(op.rg, 21015.90986, 5e-6, "R_g (nm)");
  check::near_abs(op.psi6, 0.40508, 5e-6, "psi6");
  check::near_abs(op.c6, 4.28000, 5e-6, "C6");
  check::near_abs(op.rc, 0.76363, 5e-6, "RC");

  // Per-particle field used for visualisation colouring.
  std::puts("\n-- per-particle psi6 --");
  std::vector<double> psi_local(c.np, -1.0);
  std::vector<int> nbrs(c.np, -1);
  const OrderParams op2 =
      compute_order_params_local(c, s.x.data(), s.y.data(), c.np,
                                 psi_local.data(), nbrs.data());
  check::near_abs(op2.psi6, op.psi6, 0.0,
                  "local variant returns identical globals");

  double lo = 2.0, hi = -1.0, mean = 0.0, nb_mean = 0.0;
  int nb_max = 0, isolated = 0;
  for (int i = 0; i < c.np; ++i) {
    lo = std::min(lo, psi_local[i]);
    hi = std::max(hi, psi_local[i]);
    mean += psi_local[i];
    nb_mean += nbrs[i];
    nb_max = std::max(nb_max, nbrs[i]);
    if (nbrs[i] == 0) ++isolated;
  }
  mean /= c.np;
  nb_mean /= c.np;
  check::report(lo >= 0.0 && hi <= 1.0, "every |psi6_i| lies in [0,1]",
                "range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");

  // rmin = 3780 nm = 2.634a selects the first coordination shell only, so a
  // hexatic lattice tops out near 6. Verified independently against the raw
  // coordinates: min 0, max 6, mean 4.80.
  check::near_abs(nb_mean, 4.80, 0.01, "mean neighbour count");
  check::report(nb_max <= 12, "max neighbour count is first-shell",
                "max " + std::to_string(nb_max));
  // Zero-neighbour particles are legitimate, not a defect: particle 12 sits
  // 21.2a from the centroid (R_g is 14.6a) with its nearest neighbour 2.80a
  // away, just outside the cutoff. They correctly report |psi6_i| = 0 and will
  // render at the bottom of the colour scale.
  check::eq_int(isolated, 1, "isolated particles in the golden config");
  // <|psi6_i|> must EXCEED |<psi6_i>|: local order survives averaging only if
  // the phases agree, so the phase-coherent global value is always the smaller
  // of the two. If this ever inverted, the two were confused somewhere.
  check::report(mean > op.psi6,
                "<|psi6_i|> > |<psi6_i>| (phases partially cancel)",
                "local mean " + std::to_string(mean) + " vs global " +
                    std::to_string(op.psi6));

  // RC is a pure function of (R_g, C6); check it independently of the geometry.
  check::near_abs(compute_rc(21015.90986, 4.28), 0.763633, 1e-6,
                  "RC(R_g, C6) closed form");
  // The RC formula is also self-consistent for the (different) configuration
  // quoted in the documentation, which is how we know only the config differs.
  check::near_abs(compute_rc(21014.5, 4.2667), 0.76125, 1e-5,
                  "RC reproduces the documented worked example");
}

}  // namespace

int main() {
  const Config c = Config::from_run_txt(kData + "/run.txt");
  test_run_txt_parse(c);
  test_calibration(c);
  test_mobility_table(c);
  test_order_params(c);
  return check::finish("tier1-constants");
}
