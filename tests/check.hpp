#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// Minimal assert helpers. Deliberately dependency-free -- the regression
// harness has to be buildable on a bare machine before anything else works.

namespace check {

inline int failures = 0;

inline void report(bool ok, const std::string& what, const std::string& detail) {
  std::printf("%s  %-52s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str(),
              detail.c_str());
  if (!ok) ++failures;
}

inline void eq_int(long got, long want, const std::string& what) {
  report(got == want, what,
         "got " + std::to_string(got) + ", want " + std::to_string(want));
}

inline void eq_str(const std::string& got, const std::string& want,
                   const std::string& what) {
  report(got == want, what, "got '" + got + "', want '" + want + "'");
}

// Absolute tolerance: use when the reference value is quoted to a fixed number
// of decimals (e.g. the 5-decimal columns of op1.txt).
inline void near_abs(double got, double want, double tol,
                     const std::string& what) {
  char buf[192];
  std::snprintf(buf, sizeof buf, "got %.9g, want %.9g (|d|=%.3g, tol=%.3g)", got,
                want, std::abs(got - want), tol);
  report(std::abs(got - want) <= tol, what, buf);
}

// Relative tolerance: use for derived physical quantities.
inline void near_rel(double got, double want, double rtol,
                     const std::string& what) {
  const double rel = std::abs(want) > 0 ? std::abs(got - want) / std::abs(want)
                                        : std::abs(got - want);
  char buf[192];
  std::snprintf(buf, sizeof buf, "got %.9g, want %.9g (rel=%.3g, tol=%.3g)", got,
                want, rel, rtol);
  report(rel <= rtol, what, buf);
}

inline int finish(const std::string& suite) {
  if (failures == 0) {
    std::printf("\n%s: all checks passed\n", suite.c_str());
    return 0;
  }
  std::printf("\n%s: %d CHECK(S) FAILED\n", suite.c_str(), failures);
  return 1;
}

}  // namespace check
