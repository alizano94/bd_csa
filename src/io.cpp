#include "bd_csa/io.hpp"

#include <fstream>
#include <stdexcept>

namespace bd_csa {

State read_start_txt(const std::string& path, const Config& c) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  State s(c.np);
  for (int i = 0; i < c.np; ++i) {
    double index, x, y, z;
    if (!(in >> index >> x >> y >> z))
      throw std::runtime_error(path + " truncated at particle " +
                               std::to_string(i));
    s.x[i] = x * c.a;
    s.y[i] = y * c.a;
    // z is discarded: readcn.f:39-40 overwrites it with hlev unconditionally.
  }
  return s;
}

}  // namespace bd_csa
