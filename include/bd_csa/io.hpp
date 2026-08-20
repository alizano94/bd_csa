#pragma once

#include <string>

#include "bd_csa/config.hpp"
#include "bd_csa/state.hpp"

namespace bd_csa {

// Read start.txt (readcn.f, the check == 'n' branch): four columns per line,
// (index, x, y, z), with x and y in multiples of the particle radius and scaled
// to nm on read. The z column is read and discarded -- every particle is placed
// at the fixed levitation height.
State read_start_txt(const std::string& path, const Config& c);

}  // namespace bd_csa
