#pragma once

#include <string>
#include <vector>

#include "bd_csa/config.hpp"

namespace bd_csa {

// Pre-tabulated Stokesian-dynamics mobility, D_hat(R_g, distance-from-centroid),
// relative to the bulk D0. Loaded from 2dtabledssnp300.txt: rgdssbin (30) rows
// of R_g by distdssbin (50) columns of radial distance, row-major, six columns
// per record of which only D_hat and the sample count are used.
//
// The R_g axis has a NEGATIVE stride (rgdsmin = 26500, delrgdsmin = -250), so
// row 0 is the largest R_g and rows increase as the cluster gets more compact.
class MobilityTable {
 public:
  static MobilityTable load(const std::string& path, const Config& c);

  // Nearest-bin lookup, exactly as caldss.f:21-70. Piecewise constant.
  //   rg_bin = int((Rg - rgdsmin)/delrgdsmin) + 1, clamped at >= 1
  //   unpopulated bin (count == 0) or distance out of range -> dssmax
  //   Rg beyond the last row (cluster more compact than tabulated) -> dssmin
  [[nodiscard]] double lookup_nearest(double rg_nm, double dist_nm) const;

  // C1-continuous variant: smoothstep-weighted bilinear interpolation. Needed
  // whenever the div.D drift term is enabled, because the nearest-bin table has
  // zero gradient almost everywhere and delta spikes at the bin edges.
  // Also fills grad_dist = d(D_hat)/d(distance) [1/nm] when non-null.
  [[nodiscard]] double lookup_smooth(double rg_nm, double dist_nm,
                                     double* grad_dist = nullptr) const;

  [[nodiscard]] int rows() const { return rows_; }
  [[nodiscard]] int cols() const { return cols_; }

  // Raw accessors, for tests that check the file was parsed correctly.
  [[nodiscard]] double at(int row, int col) const { return d_[idx(row, col)]; }
  [[nodiscard]] int count_at(int row, int col) const {
    return count_[idx(row, col)];
  }

 private:
  [[nodiscard]] std::size_t idx(int row, int col) const {
    return static_cast<std::size_t>(row) * cols_ + col;
  }
  // D_hat with the legacy out-of-range and unpopulated-bin rules already
  // resolved, so interpolation never straddles a hole.
  [[nodiscard]] double resolved(int row, int col) const;

  int rows_ = 0, cols_ = 0;
  double rgdsmin_ = 0, delrg_ = 0, distmin_ = 0, deldist_ = 0;
  double dssmin_ = 0, dssmax_ = 0;
  std::vector<double> d_;
  std::vector<int> count_;
};

}  // namespace bd_csa
