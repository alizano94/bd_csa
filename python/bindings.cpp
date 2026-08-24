// pybind11 bindings for bd_csa.
//
// Design rule from the plan: the simulator core knows nothing about Gym. This
// module exposes the batched simulator and the order parameters; any RL adapter
// is a thin Python layer on top (see python/bd_csa/gym_env.py).
//
// Positions cross the boundary as numpy arrays shaped (n_envs, np, 2), in
// nanometres. The legacy file interface used multiples of the particle radius
// `a`; `Config.a` is exposed so callers can convert explicitly rather than
// guessing.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "bd_csa/config.hpp"
#include "bd_csa/forces.hpp"
#include "bd_csa/io.hpp"
#include "bd_csa/mobility.hpp"
#include "bd_csa/order_params.hpp"
#include "bd_csa/simulator.hpp"
#include "bd_csa/state.hpp"

#ifdef BD_CSA_HAVE_CUDA
#include "bd_csa/sim_cuda.hpp"
#endif

namespace py = pybind11;
using namespace bd_csa;

namespace {

// Batched simulator with a CPU or CUDA backend behind one interface.
//
// The CPU path exists to validate the GPU path, and to run at all on machines
// without a device -- it is a straightforward loop over environments, not a
// performance target.
class VecSimulator {
 public:
  VecSimulator(Config cfg, const std::string& table_path, int n_envs,
               const std::string& device)
      : cfg_(std::move(cfg)),
        table_(MobilityTable::load(table_path, cfg_)),
        n_envs_(n_envs),
        device_(device) {
    if (n_envs <= 0) throw std::invalid_argument("n_envs must be positive");

    if (device_ == "cuda") {
#ifdef BD_CSA_HAVE_CUDA
      if (!SimulatorCuda::available())
        throw std::runtime_error("device='cuda' requested but no CUDA device found");
      gpu_ = std::make_unique<SimulatorCuda>(cfg_, table_, n_envs);
#else
      throw std::runtime_error("this build has no CUDA backend");
#endif
    } else if (device_ != "cpu") {
      throw std::invalid_argument("device must be 'cpu' or 'cuda'");
    }

    cpu_ = std::make_unique<SimulatorCpu>(cfg_, table_);
    states_.reserve(n_envs);
    for (int e = 0; e < n_envs; ++e) states_.emplace_back(cfg_.np);
  }

  // positions: (n_envs, np, 2) float64 in nm, or (np, 2) to broadcast one
  // configuration to every environment.
  void reset(py::array_t<double, py::array::c_style | py::array::forcecast> pos) {
    const auto b = pos.request();
    const bool broadcast = (b.ndim == 2);
    if (!broadcast && b.ndim != 3)
      throw std::invalid_argument("positions must be (n_envs, np, 2) or (np, 2)");

    const ssize_t np = broadcast ? b.shape[0] : b.shape[1];
    const ssize_t last = broadcast ? b.shape[1] : b.shape[2];
    if (np != cfg_.np || last != 2)
      throw std::invalid_argument("positions must have shape (..., np, 2) with np = " +
                                  std::to_string(cfg_.np));
    if (!broadcast && b.shape[0] != n_envs_)
      throw std::invalid_argument("positions first dimension must equal n_envs");

    const double* p = static_cast<const double*>(b.ptr);
    for (int e = 0; e < n_envs_; ++e) {
      const double* src = broadcast ? p : p + static_cast<size_t>(e) * cfg_.np * 2;
      for (int i = 0; i < cfg_.np; ++i) {
        states_[e].x[i] = src[2 * i];
        states_[e].y[i] = src[2 * i + 1];
      }
#ifdef BD_CSA_HAVE_CUDA
      if (gpu_) gpu_->upload(states_[e], e);
#endif
    }
  }

  // Advance every environment by n_steps. `lam` is either a scalar applied to
  // all environments or a per-environment array.
  //
  // NOTE the CUDA backend currently launches one lambda for the whole batch, so
  // a per-environment lambda falls back to one launch per distinct value. For
  // the common RL case (a different action per environment) this is the next
  // thing to fix -- see DEVLOG.
  void step(py::object lam, long n_steps, std::uint64_t seed) {
    std::vector<double> lambdas = to_lambda_vector(lam);

#ifdef BD_CSA_HAVE_CUDA
    if (gpu_) {
      const bool uniform = all_equal(lambdas);
      if (uniform) {
        gpu_->step(lambdas[0], n_steps, seed);
        for (int e = 0; e < n_envs_; ++e) gpu_->download(states_[e], e);
        return;
      }
      throw std::runtime_error(
          "per-environment lambda is not yet supported on the CUDA backend; "
          "use device='cpu' or a single scalar lambda");
    }
#endif
    for (int e = 0; e < n_envs_; ++e)
      cpu_->step(states_[e], lambdas[e], n_steps, seed, static_cast<std::uint32_t>(e));
  }

  // (n_envs, np, 2) float64, a copy in nm.
  [[nodiscard]] py::array_t<double> positions() const {
    py::array_t<double> out({n_envs_, cfg_.np, 2});
    auto r = out.mutable_unchecked<3>();
    for (int e = 0; e < n_envs_; ++e)
      for (int i = 0; i < cfg_.np; ++i) {
        r(e, i, 0) = states_[e].x[i];
        r(e, i, 1) = states_[e].y[i];
      }
    return out;
  }

  // (n_envs, 2) float64 = [psi6, C6/6], the RL observation.
  [[nodiscard]] py::array_t<double> observations() const {
    py::array_t<double> out({n_envs_, 2});
    auto r = out.mutable_unchecked<2>();
    for (int e = 0; e < n_envs_; ++e) {
      const OrderParams op = order_params_for(e);
      r(e, 0) = op.psi6;
      r(e, 1) = op.c6 / 6.0;
    }
    return out;
  }

  // Per-particle |psi6_i| for one environment, shape (np,), values in [0,1].
  //
  // Distinct from the global psi6, which is |<psi6_i>| -- a polycrystal of
  // well-formed grains at random orientations has high local order but low
  // global order, because the phases cancel. This is the field to colour a
  // configuration plot by.
  [[nodiscard]] py::array_t<double> local_psi6(int env) const {
    check_env(env);
    py::array_t<double> out(cfg_.np);
    std::vector<int> nb(cfg_.np);
    compute_order_params_local(cfg_, states_[env].x.data(),
                               states_[env].y.data(), cfg_.np,
                               static_cast<double*>(out.request().ptr), nb.data());
    return out;
  }

  // Neighbour count within rmin for one environment, shape (np,) int32.
  // Zero is legitimate for a particle detached from the cluster.
  [[nodiscard]] py::array_t<int> neighbour_counts(int env) const {
    check_env(env);
    py::array_t<int> out(cfg_.np);
    compute_order_params_local(cfg_, states_[env].x.data(),
                               states_[env].y.data(), cfg_.np, nullptr,
                               static_cast<int*>(out.request().ptr));
    return out;
  }

  // Full order parameters, so reward shaping can use R_g and RC without a
  // second pass over the positions.
  [[nodiscard]] std::vector<py::dict> order_parameters() const {
    std::vector<py::dict> out;
    out.reserve(n_envs_);
    for (int e = 0; e < n_envs_; ++e) {
      const OrderParams op = order_params_for(e);
      py::dict d;
      d["psi6"] = op.psi6;
      d["c6"] = op.c6;
      d["rg"] = op.rg;
      d["rc"] = op.rc;
      out.push_back(std::move(d));
    }
    return out;
  }

  [[nodiscard]] int n_envs() const { return n_envs_; }
  [[nodiscard]] const Config& config() const { return cfg_; }
  [[nodiscard]] const std::string& device() const { return device_; }

 private:
  void check_env(int e) const {
    if (e < 0 || e >= n_envs_)
      throw std::out_of_range("env index " + std::to_string(e) +
                              " outside [0, " + std::to_string(n_envs_) + ")");
  }

  [[nodiscard]] OrderParams order_params_for(int e) const {
    return compute_order_params(cfg_, states_[e].x.data(), states_[e].y.data(),
                                cfg_.np);
  }

  std::vector<double> to_lambda_vector(const py::object& lam) const {
    if (py::isinstance<py::float_>(lam) || py::isinstance<py::int_>(lam))
      return std::vector<double>(n_envs_, lam.cast<double>());
    auto arr = lam.cast<std::vector<double>>();
    if (static_cast<int>(arr.size()) != n_envs_)
      throw std::invalid_argument("lambda array must have length n_envs");
    return arr;
  }

  static bool all_equal(const std::vector<double>& v) {
    for (size_t i = 1; i < v.size(); ++i)
      if (v[i] != v[0]) return false;
    return true;
  }

  Config cfg_;
  MobilityTable table_;
  int n_envs_;
  std::string device_;
  std::unique_ptr<SimulatorCpu> cpu_;
#ifdef BD_CSA_HAVE_CUDA
  std::unique_ptr<SimulatorCuda> gpu_;
#endif
  std::vector<State> states_;
};

}  // namespace

PYBIND11_MODULE(_bd_csa, m) {
  m.doc() = "Brownian dynamics for colloidal self-assembly (C++/CUDA core)";

  py::class_<PhysicsOptions>(m, "PhysicsOptions",
                             "Physics choices that change results. Defaults are "
                             "the corrected behaviour, not the legacy one.")
      .def(py::init<>())
      .def_readwrite("mobility_update_interval",
                     &PhysicsOptions::mobility_update_interval,
                     "Steps between mobility refreshes. 1 = every step. The "
                     "legacy effectively used 'once per episode'.")
      .def_readwrite("enable_divD_drift", &PhysicsOptions::enable_divD_drift,
                     "Ito drift term for position-dependent D. Requires "
                     "smooth_mobility.")
      .def_readwrite("smooth_mobility", &PhysicsOptions::smooth_mobility,
                     "C1 interpolation of the mobility table instead of "
                     "nearest-bin lookup.")
      .def_readwrite("periodic", &PhysicsOptions::periodic,
                     "Wrap positions into the cell. Off by default: the force "
                     "loop has no minimum image.")
      .def_readwrite("continuous_overlap", &PhysicsOptions::continuous_overlap,
                     "Capped continuous repulsion instead of the legacy "
                     "discontinuous contact force.");

  py::class_<Config>(m, "Config", "Immutable simulation parameters.")
      .def(py::init<>())
      .def_static("from_run_txt", &Config::from_run_txt, py::arg("path"),
                  "Parse the legacy positional run.txt.")
      .def_readwrite("np", &Config::np)
      .def_readwrite("nstep", &Config::nstep)
      .def_readwrite("a", &Config::a, "Particle radius (nm).")
      .def_readwrite("dt", &Config::dt, "Time step (ms).")
      .def_readwrite("tempr", &Config::tempr, "Temperature (deg C).")
      .def_readwrite("dg", &Config::dg, "Electrode gap (nm).")
      .def_readwrite("rcut", &Config::rcut)
      .def_readwrite("re", &Config::re)
      .def_readwrite("rmin", &Config::rmin)
      .def_readwrite("physics", &Config::physics)
      .def_property_readonly("temperature_K", &Config::temperature_K)
      .def("__repr__", [](const Config& c) {
        return "<bd_csa.Config np=" + std::to_string(c.np) +
               " a=" + std::to_string(c.a) + "nm dt=" + std::to_string(c.dt) +
               "ms>";
      });

  py::class_<OrderParams>(m, "OrderParams")
      .def_readonly("psi6", &OrderParams::psi6)
      .def_readonly("c6", &OrderParams::c6)
      .def_readonly("rg", &OrderParams::rg)
      .def_readonly("rc", &OrderParams::rc);

  py::class_<VecSimulator>(m, "Simulator", R"doc(
Batched Brownian-dynamics simulator.

    sim = bd_csa.Simulator(config, table_path, n_envs=1024, device="cuda")
    sim.reset(x0)                       # (np, 2) broadcast, or (n_envs, np, 2)
    sim.step(lam, n_steps=1_000_000, seed=7)
    obs = sim.observations()            # (n_envs, 2) = [psi6, C6/6]

A step is a pure function of (positions, seed, lambda): there is no hidden
state, which is what makes batching sound.
)doc")
      .def(py::init<Config, const std::string&, int, const std::string&>(),
           py::arg("config"), py::arg("mobility_table"), py::arg("n_envs") = 1,
           py::arg("device") = "cpu")
      .def("reset", &VecSimulator::reset, py::arg("positions"))
      .def("step", &VecSimulator::step, py::arg("lam"), py::arg("n_steps"),
           py::arg("seed") = 0)
      .def("positions", &VecSimulator::positions)
      .def("observations", &VecSimulator::observations)
      .def("order_parameters", &VecSimulator::order_parameters)
      .def("local_psi6", &VecSimulator::local_psi6, py::arg("env") = 0,
           "Per-particle |psi6_i| in [0,1], shape (np,). This is local hexatic "
           "order, NOT the global |<psi6_i>| returned by observations(): a "
           "polycrystal has high local and low global order.")
      .def("neighbour_counts", &VecSimulator::neighbour_counts, py::arg("env") = 0,
           "Neighbours within rmin per particle, shape (np,). Zero is valid for "
           "a particle detached from the cluster.")
      .def_property_readonly("n_envs", &VecSimulator::n_envs)
      .def_property_readonly("config", &VecSimulator::config)
      .def_property_readonly("device", &VecSimulator::device);

  m.def(
      "read_start_txt",
      [](const std::string& path, const Config& c) {
        const State s = read_start_txt(path, c);
        py::array_t<double> out({c.np, 2});
        auto r = out.mutable_unchecked<2>();
        for (int i = 0; i < c.np; ++i) {
          r(i, 0) = s.x[i];
          r(i, 1) = s.y[i];
        }
        return out;
      },
      py::arg("path"), py::arg("config"),
      "Read a legacy start.txt into an (np, 2) array of nm coordinates.");

  m.def("cuda_available", [] {
#ifdef BD_CSA_HAVE_CUDA
    return SimulatorCuda::available();
#else
    return false;
#endif
  });
}
