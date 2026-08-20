// Throughput benchmark: how many environment-steps per second the GPU sustains
// as a function of batch size, against the Fortran baseline.
//
// The framing that matters: this is a THROUGHPUT win, not a latency win. A
// single environment cannot fill the GPU, so one episode will not finish much
// faster than the Fortran. Thousands of them will, in about the same wall clock.

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/sim_cuda.hpp"
#include "bd_csa/simulator.hpp"

using namespace bd_csa;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const std::string data = BD_CSA_DATA_DIR;
  const long steps = (argc > 1) ? std::atol(argv[1]) : 20000;

  const Config c = Config::from_run_txt(data + "/run.txt");
  const MobilityTable table = MobilityTable::load(data + "/2dtabledssnp300.txt", c);
  const State s0 = read_start_txt(data + "/start.txt", c);

  // Measured Fortran baseline on this machine: 6.23 s / 20000 steps.
  const double fortran_us_per_step = 311.5;

  std::printf("np = %d, steps = %ld, dt = %g ms  (%.1f s simulated per env)\n",
              c.np, steps, c.dt, steps * c.dt / 1e3);
  std::printf("Fortran baseline: %.1f us/step, 1 env\n\n", fortran_us_per_step);

  std::printf("%8s %12s %14s %16s %12s\n", "n_envs", "wall (s)", "us/env-step",
              "env-steps/s", "vs Fortran");
  std::printf("%8s %12s %14s %16s %12s\n", "------", "--------", "-----------",
              "-----------", "----------");

  if (!SimulatorCuda::available()) {
    std::printf("no CUDA device\n");
    return 1;
  }

  for (int n_envs : {1, 16, 64, 256, 1024, 2048, 4096}) {
    SimulatorCuda gpu(c, table, n_envs);
    for (int e = 0; e < n_envs; ++e) gpu.upload(s0, e);

    gpu.step(30.0, 100, 1);  // warm up / JIT-free launch

    const auto t0 = clk::now();
    gpu.step(30.0, steps, 7);
    const double wall =
        std::chrono::duration<double>(clk::now() - t0).count();

    const double env_steps = static_cast<double>(steps) * n_envs;
    const double us_per = wall * 1e6 / env_steps;
    std::printf("%8d %12.3f %14.4f %16.3e %11.1fx\n", n_envs, wall, us_per,
                env_steps / wall, fortran_us_per_step / us_per);
  }

  std::printf("\nshared memory per block: %zu bytes\n",
              SimulatorCuda(c, table, 1).shared_bytes());
  return 0;
}
