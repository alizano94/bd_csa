// Drop-in replacement for the legacy bdpd executable.
//
//   bdpd <lambda> <seed>
//
// Reads run.txt and start.txt from the working directory and writes
// out_param.json, bd_xyz<cycle>.txt and op<cycle>.txt in the legacy formats, so
// an existing driver can call this binary without changes.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "bd_csa/config.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/simulator.hpp"

using namespace bd_csa;

namespace {

// writcn.f writes coordinates back in multiples of a.
void append_coords(const std::string& path, const State& s, double a, bool truncate) {
  std::ofstream out(path, truncate ? std::ios::trunc : std::ios::app);
  for (int i = 0; i < s.size(); ++i) {
    char line[128];
    std::snprintf(line, sizeof line, "%7d%16.5f%16.5f%16.5f\n", i + 1,
                  s.x[i] / a, s.y[i] / a, 1.0663);
    out << line;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <lambda> <seed>\n", argv[0]);
    return 2;
  }
  const double lambda_arg = std::atof(argv[1]);
  const long seed_arg = std::atol(argv[2]);

  Config cfg;
  try {
    cfg = Config::from_run_txt("run.txt");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  const double lambda = lambda_arg * cfg.dpf;  // main.f:158

  // Optional flags, mainly for validation runs:
  //
  //   --mobility-interval N   0 = freeze for the whole episode, which is what
  //                           the legacy effectively did with iprint == nstep.
  //   --legacy                Reproduce the legacy physics as closely as the
  //                           new RNG allows: frozen mobility, nearest-bin
  //                           lookup, no div.D drift. Used by the statistical
  //                           comparison so that the only difference from the
  //                           Fortran is the random number stream.
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mobility-interval" && i + 1 < argc) {
      cfg.physics.mobility_update_interval = std::atol(argv[++i]);
    } else if (arg == "--legacy") {
      cfg.physics.mobility_update_interval = 0;
      cfg.physics.smooth_mobility = false;
      cfg.physics.enable_divD_drift = false;
      cfg.physics.continuous_overlap = false;
    }
  }

  const MobilityTable table = MobilityTable::load(cfg.rgdsfile, cfg);
  State s = read_start_txt(cfg.par_in, cfg);

  SimulatorCpu sim(cfg, table);

  const std::string traj = cfg.par_out + "1.txt";
  const std::string oplog = "op1.txt";

  // t = 0 snapshot, then the episode.
  OrderParams op = sim.order_params(s);
  append_coords(traj, s, cfg.a, /*truncate=*/true);
  {
    std::ofstream f(oplog, std::ios::trunc);
    char line[256];
    std::snprintf(line, sizeof line, "%12.5f%12.5f%15.5f%12.5f%12.5f\n", 0.0,
                  op.c6, op.rg, op.psi6, op.rc);
    f << line;
  }

  sim.step(s, lambda, cfg.nstep, static_cast<std::uint64_t>(seed_arg));

  op = sim.order_params(s);
  append_coords(traj, s, cfg.a, /*truncate=*/false);
  {
    std::ofstream f(oplog, std::ios::app);
    char line[256];
    std::snprintf(line, sizeof line, "%12.5f%12.5f%15.5f%12.5f%12.5f\n",
                  cfg.nstep * cfg.dt / 1e3, op.c6, op.rg, op.psi6, op.rc);
    f << line;
  }

  // The RL driver chains this seed into the next episode, so it must evolve.
  // With a counter-based RNG there is no evolving generator state to report;
  // derive the next seed deterministically from (seed, steps) instead.
  const std::uint64_t next_seed =
      (static_cast<std::uint64_t>(seed_arg) * 6364136223846793005ULL + 1442695040888963407ULL) >> 33;

  std::ofstream json("out_param.json", std::ios::trunc);
  json << " {\n";
  json << "       \"step\":       " << cfg.nstep << " ,\n";
  json << "       \"lambda\":   " << lambda << "      ,\n";
  json << "       \"psi6\":  " << op.psi6 << "      ,\n";
  json << "       \"c6\":   " << op.c6 << "      ,\n";
  json << "       \"seed\":  " << next_seed << "\n";
  json << " }\n";
  return 0;
}
