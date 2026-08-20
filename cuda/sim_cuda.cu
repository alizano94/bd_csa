#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd_csa/sim_cuda.hpp"
#include "kernels.cuh"

namespace bd_csa {
namespace {

using cuda::DevConfig;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// One thread per particle, rounded up to a whole warp.
//
// This was 256, chosen so the reduction tree could assume a power of two. That
// was a bad trade: with np = 300 the strided loop gave threads 0..43 two
// particles and everyone else one, so every block ran at the pace of the
// double-loaded threads and the force loop wasted ~50% of its time. At 320,
// each thread owns exactly one particle and only the 20 tail threads idle.
constexpr int kBlock = 320;

// Verlet neighbour-list capacity, per particle.
//
// Measured on the golden configuration: with a list radius of 5.5a (the 5a force
// cutoff plus a 0.5a skin) each particle has a mean of 16.3 and a maximum of 21
// neighbours. 32 leaves ~50% headroom for the cluster densifying during
// annealing, and overflow is handled correctly rather than silently truncated
// (see kOverflow below).
constexpr int kMaxNb = 32;

// Sentinel stored in the neighbour count when a particle exceeded kMaxNb. Such
// particles fall back to a full O(N) scan for that step, so the physics stays
// exact no matter how dense the cluster gets.
constexpr unsigned short kOverflow = 0xFFFFu;

// How often to test whether the lists have gone stale. Displacement is diffusive
// at ~2e-3 a/step, so 64 steps moves a particle ~0.016a -- far below the 0.25a
// half-skin that would invalidate a list. Checking every step would pay for two
// block reductions to save a rebuild that is only needed every ~15,000 steps.
constexpr int kRebuildCheck = 64;

// Block-wide max, same structure as block_sum.
__device__ __forceinline__ double block_max(double v, double* red, int tid) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmax(v, __shfl_down_sync(0xffffffffu, v, off));
  const int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (tid == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    double m = red[0];
    for (int i = 1; i < nwarps; ++i) m = fmax(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  const double r = red[0];
  __syncthreads();
  return r;
}

// Block-wide sum, valid for any block size. Warp-level shuffle reduction, then
// one pass over the (at most 32) per-warp partials.
__device__ __forceinline__ double block_sum(double v, double* red, int tid) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);

  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) red[warp] = v;
  __syncthreads();

  if (tid == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    double s = 0.0;
    for (int i = 0; i < nwarps; ++i) s += red[i];
    red[0] = s;
  }
  __syncthreads();
  const double r = red[0];
  __syncthreads();
  return r;
}

// One block per environment. Positions live in shared memory for the entire
// episode, so the hot loop touches global memory exactly twice: once to load
// and once to store. At np = 300 the whole system is 4.8 kB of doubles, and
// there is no cross-block communication -- every reduction the physics needs
// (centroid, R_g) is per environment, hence per block.
//
// This also means the 10^6 integration steps run in a SINGLE kernel launch;
// launching per step would cost ~5 s of pure launch overhead alone.
__global__ __launch_bounds__(kBlock) void episode_kernel(
    DevConfig c, const float* __restrict__ mobtab, double* __restrict__ gx,
    double* __restrict__ gy, float lambda, long nsteps,
    unsigned long long seed, float* __restrict__ fout) {
  extern __shared__ double smem[];
  const int np = c.np;
  double* sx = smem;
  double* sy = smem + np;
  double* red = smem + 2 * np;
  float* fx = reinterpret_cast<float*>(red + kBlock);
  float* fy = fx + np;

  const int env = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(env) * np;

  for (int i = tid; i < np; i += kBlock) {
    sx[i] = gx[base + i];
    sy[i] = gy[base + i];
  }
  __syncthreads();

  // Per-particle mobility, refreshed on the schedule in the config. Held in
  // shared memory so a thread servicing several particles keeps them all.
  float* mob = fy + np;
  float* gdx = mob + np;
  float* gdy = gdx + np;
  // Positions as of the last neighbour-list build, in FP32. Used only to decide
  // when a list has gone stale, so single precision is ample.
  float* bx = gdy + np;
  float* by = bx + np;
  // Neighbour lists: kMaxNb indices per particle, plus a per-particle count.
  unsigned short* nbr = reinterpret_cast<unsigned short*>(by + np);
  unsigned short* nnb = nbr + np * kMaxNb;

  for (long step = 0; step < nsteps; ++step) {
    if (c.mob_interval > 0 ? (step % c.mob_interval == 0) : (step == 0)) {
      double sumx = 0.0, sumy = 0.0;
      for (int i = tid; i < np; i += kBlock) {
        sumx += sx[i];
        sumy += sy[i];
      }
      const double cx = block_sum(sumx, red, tid) / np;
      const double cy = block_sum(sumy, red, tid) / np;

      double var = 0.0;
      for (int i = tid; i < np; i += kBlock) {
        const double dx = sx[i] - cx, dy = sy[i] - cy;
        var += dx * dx + dy * dy;
      }
      const double rg = sqrt(block_sum(var, red, tid) / np);

      for (int i = tid; i < np; i += kBlock) {
        const double dx = sx[i] - cx, dy = sy[i] - cy;
        const double dist = sqrt(dx * dx + dy * dy);
        float grad = 0.0f;
        mob[i] = cuda::mob_lookup(c, mobtab, rg, dist, &grad);
        if (c.enable_drift && dist > 0.0) {
          gdx[i] = grad * (float)(dx / dist);
          gdy[i] = grad * (float)(dy / dist);
        } else {
          gdx[i] = 0.0f;
          gdy[i] = 0.0f;
        }
      }
      __syncthreads();
    }

    // --- Verlet neighbour lists -------------------------------------------
    // Only ~16 of the 299 partners lie inside the 5a force cutoff, a 20x
    // redundancy. Restricting the loop to a neighbour list is what turns that
    // redundancy into speed: an earlier attempt that merely *rejected* far pairs
    // cheaply inside the full loop gained only 14%, because the cost was the
    // 299 loop iterations themselves, not the work inside them.
    bool rebuild = (step == 0);
    if (!rebuild && (step % kRebuildCheck) == 0) {
      double d2 = 0.0;
      for (int i = tid; i < np; i += kBlock) {
        const double ddx = sx[i] - bx[i], ddy = sy[i] - by[i];
        d2 = fmax(d2, ddx * ddx + ddy * ddy);
      }
      // A list stays valid until a particle moves half the skin: two particles
      // approaching each other can each move that far before a pair that was
      // outside the list radius could enter the force cutoff.
      rebuild = block_max(d2, red, tid) > c.skin_half2;
    }
    if (rebuild) {
      for (int i = tid; i < np; i += kBlock) {
        bx[i] = (float)sx[i];
        by[i] = (float)sy[i];
      }
      __syncthreads();
      for (int i = tid; i < np; i += kBlock) {
        const float xi = bx[i], yi = by[i];
        int n = 0;
        for (int j = 0; j < np; ++j) {
          if (j == i) continue;
          const float ddx = bx[j] - xi, ddy = by[j] - yi;
          if (ddx * ddx + ddy * ddy <= c.rlist2) {
            if (n < kMaxNb) nbr[i * kMaxNb + n] = (unsigned short)j;
            ++n;
          }
        }
        nnb[i] = (n <= kMaxNb) ? (unsigned short)n : kOverflow;
      }
      __syncthreads();
    }

    // --- forces -----------------------------------------------------------
    for (int i = tid; i < np; i += kBlock) {
      // Plain FP32 accumulation. Kahan compensated summation was tried here on
      // the theory that the partially-cancelling pair sum was amplifying
      // rounding; measured, it changed the RMS force error by 0.1%
      // (9.798e-6 -> 9.812e-6). The budget is dominated by rounding within each
      // pair term, so the compensation was pure cost and was removed.
      float ax = 0.0f, ay = 0.0f;
      const double xi = sx[i], yi = sy[i];

      // Overflowed particles scan everything, so a list that could not hold all
      // neighbours degrades performance rather than correctness.
      const unsigned short cnt = nnb[i];
      const bool overflow = (cnt == kOverflow);
      const int niter = overflow ? np : (int)cnt;

      for (int k = 0; k < niter; ++k) {
        const int j = overflow ? k : (int)nbr[i * kMaxNb + k];
        if (j == i) continue;

        // Accurate FP64 difference: see the precision note in kernels.cuh --
        // the DLVO exponential's 0.1/nm sensitivity makes an FP32 difference of
        // ~2e4 nm coordinates unusable. pair_force_on_self still applies the
        // exact cutoffs, so list members beyond 5a contribute exactly zero.
        const float dx = (float)(sx[j] - xi);
        const float dy = (float)(sy[j] - yi);
        float pfx, pfy;
        cuda::pair_force_on_self(c, lambda, dx, dy, (float)xi, (float)yi,
                                 (float)sx[j], (float)sy[j], i < j, pfx, pfy);
        ax += pfx;
        ay += pfy;
      }

      float dex, dey;
      cuda::grad_emag_sq_f(c, (float)xi, (float)yi, dex, dey);
      ax += (float)c.Fdep_coef * lambda * dex;
      ay += (float)c.Fdep_coef * lambda * dey;
      fx[i] = ax;
      fy[i] = ay;
    }
    __syncthreads();  // forces complete before any position moves

    if (fout) {  // force-probe mode: one evaluation, then out
      for (int i = tid; i < np; i += kBlock) {
        fout[2 * (base + i)] = fx[i];
        fout[2 * (base + i) + 1] = fy[i];
      }
      return;
    }

    // --- position update, FP64 accumulator --------------------------------
    for (int i = tid; i < np; i += kBlock) {
      const double D = mob[i];
      float zx, zy;
      cuda::philox_normal2(seed, env, i, (unsigned long long)step, zx, zy);
      const double sqrtD = sqrt(D);
      double ddx = D * fx[i] * c.fac1_dt + sqrtD * zx * c.fac2_dt;
      double ddy = D * fy[i] * c.fac1_dt + sqrtD * zy * c.fac2_dt;
      if (c.enable_drift) {
        ddx += c.drift_coef * gdx[i];
        ddy += c.drift_coef * gdy[i];
      }
      sx[i] += ddx;
      sy[i] += ddy;
      if (c.periodic) {
        sx[i] -= c.dg * nearbyint(sx[i] / c.dg);
        sy[i] -= c.dg * nearbyint(sy[i] / c.dg);
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < np; i += kBlock) {
    gx[base + i] = sx[i];
    gy[base + i] = sy[i];
  }
}

}  // namespace

struct SimulatorCuda::Impl {
  DevConfig dc{};
  int n_envs = 0;
  int np = 0;
  double* d_x = nullptr;
  double* d_y = nullptr;
  float* d_mob = nullptr;
  float* d_f = nullptr;
  size_t shmem = 0;
};

SimulatorCuda::SimulatorCuda(const Config& cfg, const MobilityTable& table,
                             int n_envs)
    : impl_(new Impl), cfg_(cfg) {
  impl_->n_envs = n_envs;
  impl_->np = cfg.np;

  DevConfig& d = impl_->dc;
  d.np = cfg.np;
  d.mob_rows = table.rows();
  d.mob_cols = table.cols();
  d.a = cfg.a;
  d.dg = cfg.dg;
  d.re = cfg.re;
  d.rcut = cfg.rcut;
  d.kappa = cfg.kappa;
  d.pfpp = cfg.pfpp;
  d.fcm = cfg.fcm;
  d.Fhw = cfg.Fhw;

  const double kT = cfg.kb * cfg.temperature_K();
  d.Fo_coef = 1e18 * 0.75 * kT / cfg.a;
  d.Fpp_pref = 1e18 * kT * cfg.kappa * cfg.pfpp / cfg.a;
  d.Fdep_coef = 2.0 * 1e18 * kT / cfg.fcm;

  // Cutoff for the FP32 rejection test, with 1 nm of slack to cover the ~2.4e-3
  // nm error of single-precision shadow positions. Slack only admits extra
  // pairs, so it cannot change the physics.
  {
    const double cut = (cfg.rcut > cfg.re ? cfg.rcut : cfg.re) + 1.0;
    d.cut2_margin = (float)(cut * cut);
    // Verlet list radius = force cutoff + skin. The skin trades list size
    // against rebuild frequency; 0.5a gives 16 neighbours on average and a
    // rebuild roughly every 15,000 steps.
    const double skin = 0.5 * cfg.a;
    const double rlist = (cfg.rcut > cfg.re ? cfg.rcut : cfg.re) + skin;
    d.rlist2 = (float)(rlist * rlist);
    d.skin_half2 = 0.25 * skin * skin;
  }
  d.inv_dg = (float)(1.0 / cfg.dg);
  d.inv_a = (float)(1.0 / cfg.a);
  d.fac1_dt = cfg.fac1 * cfg.dt;
  d.fac2_dt = cfg.fac2 * cfg.dt;
  d.drift_coef = 0.5 * d.fac2_dt * d.fac2_dt;

  d.rgdsmin = cfg.rgdsmin;
  d.delrg = cfg.delrgdsmin;
  d.distmin = cfg.distmin;
  d.deldist = cfg.deldist;
  d.dssmin = cfg.dssmin;
  d.dssmax = cfg.dssmax;
  d.ecorrectflag = cfg.ecorrectflag;
  d.smooth_mobility = cfg.physics.smooth_mobility;
  d.enable_drift = cfg.physics.enable_divD_drift && cfg.physics.smooth_mobility;
  d.periodic = cfg.physics.periodic;
  d.legacy_overlap = 0;
  d.mob_interval = cfg.physics.mobility_update_interval;

  // Pre-resolve the unpopulated-bin rule on the host so the device never has to
  // consult the sample counts.
  std::vector<float> flat(static_cast<size_t>(table.rows()) * table.cols());
  for (int r = 0; r < table.rows(); ++r)
    for (int cc = 0; cc < table.cols(); ++cc)
      flat[static_cast<size_t>(r) * table.cols() + cc] =
          static_cast<float>(table.count_at(r, cc) >= 1 ? table.at(r, cc)
                                                        : cfg.dssmax);

  const size_t n = static_cast<size_t>(n_envs) * cfg.np;
  check(cudaMalloc(&impl_->d_x, n * sizeof(double)), "cudaMalloc x");
  check(cudaMalloc(&impl_->d_y, n * sizeof(double)), "cudaMalloc y");
  check(cudaMalloc(&impl_->d_mob, flat.size() * sizeof(float)), "cudaMalloc mob");
  check(cudaMalloc(&impl_->d_f, 2 * n * sizeof(float)), "cudaMalloc f");
  check(cudaMemcpy(impl_->d_mob, flat.data(), flat.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "upload mobility table");

  // sx, sy (FP64) | reduction scratch | fx, fy, mob, gdx, gdy, bx, by (FP32)
  // | neighbour lists + counts (uint16)
  impl_->shmem = 2 * cfg.np * sizeof(double) + kBlock * sizeof(double) +
                 7 * cfg.np * sizeof(float) +
                 cfg.np * (kMaxNb + 1) * sizeof(unsigned short);
}

SimulatorCuda::~SimulatorCuda() {
  if (impl_) {
    cudaFree(impl_->d_x);
    cudaFree(impl_->d_y);
    cudaFree(impl_->d_mob);
    cudaFree(impl_->d_f);
    delete impl_;
  }
}

bool SimulatorCuda::available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

size_t SimulatorCuda::shared_bytes() const { return impl_->shmem; }

void SimulatorCuda::upload(const State& s, int env) {
  const size_t off = static_cast<size_t>(env) * impl_->np;
  check(cudaMemcpy(impl_->d_x + off, s.x.data(), impl_->np * sizeof(double),
                   cudaMemcpyHostToDevice), "upload x");
  check(cudaMemcpy(impl_->d_y + off, s.y.data(), impl_->np * sizeof(double),
                   cudaMemcpyHostToDevice), "upload y");
}

void SimulatorCuda::download(State& s, int env) const {
  const size_t off = static_cast<size_t>(env) * impl_->np;
  check(cudaMemcpy(s.x.data(), impl_->d_x + off, impl_->np * sizeof(double),
                   cudaMemcpyDeviceToHost), "download x");
  check(cudaMemcpy(s.y.data(), impl_->d_y + off, impl_->np * sizeof(double),
                   cudaMemcpyDeviceToHost), "download y");
}

void SimulatorCuda::step(double lambda, long n_steps, std::uint64_t seed) {
  episode_kernel<<<impl_->n_envs, kBlock, impl_->shmem>>>(
      impl_->dc, impl_->d_mob, impl_->d_x, impl_->d_y, static_cast<float>(lambda),
      n_steps, seed, nullptr);
  check(cudaGetLastError(), "episode_kernel launch");
  check(cudaDeviceSynchronize(), "episode_kernel");
}

std::vector<float> SimulatorCuda::forces_once(double lambda, bool legacy_overlap) {
  DevConfig d = impl_->dc;
  d.legacy_overlap = legacy_overlap;
  d.mob_interval = 0;
  episode_kernel<<<impl_->n_envs, kBlock, impl_->shmem>>>(
      d, impl_->d_mob, impl_->d_x, impl_->d_y, static_cast<float>(lambda), 1, 0,
      impl_->d_f);
  check(cudaGetLastError(), "force kernel launch");
  check(cudaDeviceSynchronize(), "force kernel");

  std::vector<float> out(2 * static_cast<size_t>(impl_->n_envs) * impl_->np);
  check(cudaMemcpy(out.data(), impl_->d_f, out.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "download forces");
  return out;
}

}  // namespace bd_csa
