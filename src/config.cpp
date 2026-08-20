#include "bd_csa/config.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bd_csa {
namespace {

// run.txt is a positional label/value file read with Fortran list-directed
// input. Every read(1,*) here consumes exactly one record, so "one read = one
// line" is an exact model of main.f:74-156.
class LineReader {
 public:
  explicit LineReader(const std::string& path) : in_(path), path_(path) {
    if (!in_) throw std::runtime_error("cannot open " + path);
  }

  void skip() { (void)next(); }

  std::string raw() { return next(); }

  // Read the first sizeof...(T) whitespace-separated tokens of one record.
  // Trailing tokens on the record are discarded, matching Fortran (e.g. the
  // expbox line carries six values but only two are read).
  template <typename... T>
  void read(T&... out) {
    std::istringstream ss(next());
    (extract(ss, out), ...);
  }

 private:
  std::string next() {
    std::string line;
    if (!std::getline(in_, line))
      throw std::runtime_error("unexpected end of " + path_ + " at line " +
                               std::to_string(line_no_));
    ++line_no_;
    return line;
  }

  template <typename T>
  static void extract(std::istringstream& ss, T& out) {
    if constexpr (std::is_same_v<T, std::string>) {
      ss >> out;
      strip_quotes(out);
    } else if constexpr (std::is_same_v<T, char>) {
      std::string tok;
      ss >> tok;
      strip_quotes(tok);
      out = tok.empty() ? ' ' : tok[0];
    } else {
      ss >> out;
    }
  }

  static void strip_quotes(std::string& s) {
    if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"') &&
        s.back() == s.front())
      s = s.substr(1, s.size() - 2);
  }

  std::ifstream in_;
  std::string path_;
  int line_no_ = 1;
};

}  // namespace

double Config::force_scale() const {
  return 1e18 * kb * temperature_K() / a;
}

double Config::D0_nm2_per_s(double eta_Pa_s) const {
  const double a_m = a * 1e-9;
  const double D0_m2_per_s = kb * temperature_K() / (6.0 * M_PI * eta_Pa_s * a_m);
  return D0_m2_per_s * 1e18;  // m^2/s -> nm^2/s
}

Config Config::from_run_txt(const std::string& path) {
  Config c;
  LineReader r(path);

  // Discarded on these lines: a second calibration value on the fac1/fac2
  // records (the particle-wall variants), four extra values on the expbox
  // record, and the histogram parameters (dead code, 7.10).
  std::string dead_s;
  double dead_d;

  r.skip(); r.read(c.np);
  r.skip(); r.read(c.nstep);
  r.skip(); r.read(c.iprint);
  r.skip(); r.read(c.istart);
  r.skip(); r.read(c.par_in);
  r.skip(); r.read(c.par_out);
  r.skip(); r.read(c.a);
  r.skip(); r.read(c.tempr);
  r.skip(); r.read(c.phi);
  r.skip(); r.read(c.dt);
  r.skip(); r.read(c.t0);
  r.skip(); r.read(c.check);
  r.skip(); r.read(c.fac1_raw);
  r.skip(); r.read(c.fac2_raw);
  r.skip(); r.read(c.pwfactor);
  r.skip(); r.read(c.Fgrav);
  r.skip(); r.read(c.rcut);
  r.skip(); r.read(c.re);
  r.skip(); r.read(c.kappa);
  r.skip(); r.read(c.pfpp);
  r.skip(); r.read(c.pfpw);
  r.skip(); r.skip();          // "lambda" (lines 43-44) -- comes from argv
  r.skip(); r.read(c.fcm);
  r.skip(); r.read(c.dg);
  r.skip(); r.read(c.hlev);
  r.skip(); r.read(c.rmin);
  r.skip(); r.read(c.expbox[0], c.expbox[1]);
  r.skip(); r.read(c.var);
  r.skip(); r.read(c.polymono);
  r.skip(); r.read(c.pdfile);
  r.skip(); r.skip();          // "idummy" (lines 61-62) -- comes from argv
  r.skip(); r.read(dead_s, dead_d, dead_d);   // RGHISTFILE  (dead)
  r.skip(); r.read(dead_s, dead_d);           // PSIHISTFILE (dead)
  r.skip(); r.read(dead_s, dead_d);           // conhistfile (dead)
  r.skip(); r.read(c.cyclenum);
  r.skip(); c.rgdsfile = r.raw();             // read(1,'(a150)')
  r.skip(); r.read(c.rgdsmin, c.delrgdsmin, c.rgdssbin);
  r.skip(); r.read(c.distmin, c.deldist, c.distdssbin);
  r.skip(); r.read(c.dssmin, c.dssmax);
  r.skip(); r.read(c.dpf);
  r.skip(); r.read(c.ecorrectflag);

  // Trim the fixed-width rgdsfile record.
  const auto end = c.rgdsfile.find_last_not_of(" \t\r\n");
  c.rgdsfile = (end == std::string::npos) ? "" : c.rgdsfile.substr(0, end + 1);

  // Rescalings, main.f:170-177.
  c.pfpp *= c.a;
  c.rcut *= c.a;
  c.re   *= c.a;
  c.dg   *= c.a;
  c.hlev  = c.a + c.a * c.hlev;
  c.expbox[0] *= c.a;
  c.expbox[1] *= c.a;

  // Calibration rescalings, main.f:223-225.
  c.fac1 = c.fac1_raw / c.a;
  c.fac2 = c.fac2_raw * std::sqrt(c.temperature_K() / c.a) / std::sqrt(c.dt);

  return c;
}

}  // namespace bd_csa
