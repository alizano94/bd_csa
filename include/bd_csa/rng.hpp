#pragma once

#include <cmath>
#include <cstdint>

namespace bd_csa {

// Counter-based RNG (Philox 4x32-10, Random123 / cuRAND compatible).
//
// Replaces the legacy ran2 + gasdev, which had two disqualifying properties:
//
//  1. ran2 only initialises its shuffle table when the seed is <= 0, but the
//     driver always passes a non-negative seed. Every positive seed therefore
//     produced byte-identical first 8 deviates -- verified: seeds 1, 7, 999 and
//     123456 all start 0.65541640, 0.20099520, ... (07-porting-notes.md 7.1).
//  2. gasdev caches its second Box-Muller deviate in a SAVEd variable and its
//     rejection loop consumes a variable number of uniforms, so there is no
//     fixed mapping from step number to stream position -- unusable on a GPU.
//
// Being counter-based, this is stateless: the value is a pure function of the
// key, so any (particle, step, component) can be drawn independently and in any
// order. That gives bitwise reproducibility regardless of thread scheduling,
// which is exactly what the batched CUDA backend needs.
class Philox {
 public:
  // Draw two standard normals for the given coordinate in the stream.
  static void normal2(std::uint64_t seed, std::uint32_t env, std::uint32_t particle,
                      std::uint64_t step, double& z0, double& z1) {
    std::uint32_t ctr[4] = {
        static_cast<std::uint32_t>(step), static_cast<std::uint32_t>(step >> 32),
        particle, env};
    std::uint32_t key[2] = {static_cast<std::uint32_t>(seed),
                            static_cast<std::uint32_t>(seed >> 32)};
    round10(ctr, key);
    box_muller(ctr, z0, z1);
  }

 private:
  static constexpr std::uint32_t kW0 = 0x9E3779B9u;  // golden ratio
  static constexpr std::uint32_t kW1 = 0xBB67AE85u;  // sqrt(3) - 1
  static constexpr std::uint32_t kM0 = 0xD2511F53u;
  static constexpr std::uint32_t kM1 = 0xCD9E8D57u;

  static void mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi,
                      std::uint32_t& lo) {
    const std::uint64_t p = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(p >> 32);
    lo = static_cast<std::uint32_t>(p);
  }

  static void round10(std::uint32_t* c, std::uint32_t* k) {
    for (int i = 0; i < 10; ++i) {
      std::uint32_t hi0, lo0, hi1, lo1;
      mulhilo(kM0, c[0], hi0, lo0);
      mulhilo(kM1, c[2], hi1, lo1);
      const std::uint32_t n0 = hi1 ^ c[1] ^ k[0];
      const std::uint32_t n1 = lo1;
      const std::uint32_t n2 = hi0 ^ c[3] ^ k[1];
      const std::uint32_t n3 = lo0;
      c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
      k[0] += kW0;
      k[1] += kW1;
    }
  }

  // Box-Muller. Unlike gasdev's polar-rejection form this consumes a fixed
  // number of uniforms, so stream position stays a pure function of the key.
  static void box_muller(const std::uint32_t* c, double& z0, double& z1) {
    // Open interval (0,1): avoid log(0).
    constexpr double kScale = 2.3283064365386963e-10;  // 2^-32
    const double u1 = (static_cast<double>(c[0]) + 0.5) * kScale;
    const double u2 = (static_cast<double>(c[1]) + 0.5) * kScale;
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;
    z0 = r * std::cos(theta);
    z1 = r * std::sin(theta);
  }
};

}  // namespace bd_csa
