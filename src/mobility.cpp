#include "bd_csa/mobility.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace bd_csa {

MobilityTable MobilityTable::load(const std::string& path, const Config& c) {
  MobilityTable t;
  t.rows_ = c.rgdssbin;
  t.cols_ = c.distdssbin;
  t.rgdsmin_ = c.rgdsmin;
  t.delrg_ = c.delrgdsmin;
  t.distmin_ = c.distmin;
  t.deldist_ = c.deldist;
  t.dssmin_ = c.dssmin;
  t.dssmax_ = c.dssmax;
  t.d_.assign(static_cast<std::size_t>(t.rows_) * t.cols_, 0.0);
  t.count_.assign(t.d_.size(), 0);

  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open mobility table " + path);

  // main.f:161-166 -- i over rows (R_g) outer, j over columns (distance) inner.
  // The first four fields (bin indices and bin centres) are read into throwaway
  // variables; binning is recomputed from run.txt, not taken from the file.
  for (int i = 0; i < t.rows_; ++i) {
    for (int j = 0; j < t.cols_; ++j) {
      double dum1, dum2, rg_centre, dist_centre, dhat;
      int count;
      if (!(in >> dum1 >> dum2 >> rg_centre >> dist_centre >> dhat >> count))
        throw std::runtime_error("mobility table truncated at row " +
                                 std::to_string(i) + " col " +
                                 std::to_string(j));
      t.d_[t.idx(i, j)] = dhat;
      t.count_[t.idx(i, j)] = count;
    }
  }
  return t;
}

double MobilityTable::resolved(int row, int col) const {
  if (row >= rows_) return dssmin_;
  row = std::clamp(row, 0, rows_ - 1);
  if (col < 0 || col >= cols_) return dssmax_;
  return count_[idx(row, col)] >= 1 ? d_[idx(row, col)] : dssmax_;
}

double MobilityTable::lookup_nearest(double rg_nm, double dist_nm) const {
  // caldss.f:21 -- Fortran int() truncates toward zero, and the +1 makes the
  // index 1-based; we shift to 0-based.
  int rg_bin = static_cast<int>((rg_nm - rgdsmin_) / delrg_) + 1;
  if (rg_bin <= 0) rg_bin = 1;
  const int dist_bin = static_cast<int>((dist_nm - distmin_) / deldist_) + 1;

  if (rg_bin >= 1 && rg_bin <= rows_) {
    if (dist_bin >= 1 && dist_bin <= cols_) {
      return count_[idx(rg_bin - 1, dist_bin - 1)] >= 1
                 ? d_[idx(rg_bin - 1, dist_bin - 1)]
                 : dssmax_;
    }
    return dssmax_;
  }
  return dssmin_;  // rg_bin > rows_: more compact than tabulated
}

double MobilityTable::lookup_smooth(double rg_nm, double dist_nm,
                                    double* grad_dist) const {
  // Continuous bin coordinates measured from bin centres, so that the
  // interpolant reproduces the tabulated value at each centre.
  const double rg_x = (rg_nm - rgdsmin_) / delrg_ - 0.5;
  const double dist_x = (dist_nm - distmin_) / deldist_ - 0.5;

  const int r0 = static_cast<int>(std::floor(rg_x));
  const int c0 = static_cast<int>(std::floor(dist_x));
  const double fr = rg_x - r0;
  const double fc = dist_x - c0;

  // smoothstep: C1 at the bin edges (its derivative vanishes there), unlike
  // plain bilinear which leaves grad.D discontinuous.
  const auto smooth = [](double x) { return x * x * (3.0 - 2.0 * x); };
  const auto dsmooth = [](double x) { return 6.0 * x * (1.0 - x); };

  const double wr = smooth(fr), wc = smooth(fc);

  const double v00 = resolved(r0, c0);
  const double v01 = resolved(r0, c0 + 1);
  const double v10 = resolved(r0 + 1, c0);
  const double v11 = resolved(r0 + 1, c0 + 1);

  const double lo = v00 + (v01 - v00) * wc;
  const double hi = v10 + (v11 - v10) * wc;
  const double value = lo + (hi - lo) * wr;

  if (grad_dist) {
    // d(D_hat)/d(dist) = d/dwc * dwc/dfc * dfc/d(dist)
    const double dlo = (v01 - v00) * dsmooth(fc);
    const double dhi = (v11 - v10) * dsmooth(fc);
    *grad_dist = (dlo + (dhi - dlo) * wr) / deldist_;
  }
  return value;
}

}  // namespace bd_csa
