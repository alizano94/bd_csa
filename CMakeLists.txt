cmake_minimum_required(VERSION 3.24)
project(bd_csa LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The static core and CUDA libraries get linked into the Python extension, which
# is a shared object -- every translation unit inside it must be position
# independent or the link fails with "relocation R_X86_64_PC32 ... can not be
# used when making a shared object".
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# CUDA is opt-in-if-available; the CPU reference must always build standalone
# so the test suite runs on machines without a GPU.
include(CheckLanguage)
check_language(CUDA)
# Compute the default explicitly: passing the compiler path straight into
# option() coerces to OFF, since a path is not a boolean constant.
if(CMAKE_CUDA_COMPILER)
  set(_bd_csa_cuda_default ON)
else()
  set(_bd_csa_cuda_default OFF)
endif()
option(BD_CSA_ENABLE_CUDA "Build the CUDA backend" ${_bd_csa_cuda_default})
if(BD_CSA_ENABLE_CUDA)
  # Must be set BEFORE enable_language(CUDA): enable_language picks a default
  # (here sm_75) and a later assignment is ignored.
  #
  # "-real" emits SASS only, with no embedded PTX. That is deliberate. Shipping
  # PTX makes the driver JIT it at module load, and JIT requires
  # driver >= toolkit; this machine pairs a CUDA 13.2 toolkit with a driver that
  # supports 13.0, so any PTX path fails with "unsupported toolchain". Native
  # SASS for the local architecture sidesteps it entirely.
  #
  # Override with -DBD_CSA_CUDA_ARCH=<arch> on a different GPU (e.g. 86-real).
  set(BD_CSA_CUDA_ARCH "89-real" CACHE STRING "CUDA architecture (default: Ada, cc 8.9)")
  set(CMAKE_CUDA_ARCHITECTURES ${BD_CSA_CUDA_ARCH} CACHE STRING "" FORCE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(bd_csa_core
  src/config.cpp
  src/mobility.cpp
  src/order_params.cpp
  src/forces.cpp
  src/simulator.cpp
  src/io.cpp
)
target_include_directories(bd_csa_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(bd_csa_core PRIVATE -Wall -Wextra)

# Point tests at the frozen golden inputs rather than the pristine legacy tree.
target_compile_definitions(bd_csa_core PUBLIC
  BD_CSA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  BD_CSA_BUILD_DIR="${CMAKE_CURRENT_SOURCE_DIR}/build")

if(BD_CSA_ENABLE_CUDA)
  add_library(bd_csa_cuda cuda/sim_cuda.cu)
  target_include_directories(bd_csa_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                         PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
  target_link_libraries(bd_csa_cuda PUBLIC bd_csa_core)
  # Separable compilation is deliberately OFF. All device code lives in the
  # single translation unit sim_cuda.cu (kernels.cuh is header-only), so there
  # is nothing to device-link across objects -- and leaving it on emits an
  # unresolved device-link stub that surfaces as
  # "undefined symbol: fatbinData" when this static library is pulled into the
  # Python extension's shared object.
  set_target_properties(bd_csa_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_definitions(bd_csa_core PUBLIC BD_CSA_HAVE_CUDA=1)

  add_executable(bench_cuda apps/bench_cuda.cpp)
  target_link_libraries(bench_cuda PRIVATE bd_csa_cuda)
endif()

add_executable(bdpd apps/bdpd_main.cpp)
target_link_libraries(bdpd PRIVATE bd_csa_core)

# ---- Python bindings (optional) --------------------------------------------
# pybind11 is fetched rather than required from the system: this machine has no
# pip, and fetching keeps the build self-contained. Turn off with
# -DBD_CSA_BUILD_PYTHON=OFF when offline.
option(BD_CSA_BUILD_PYTHON "Build the Python extension module" ON)
if(BD_CSA_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(Python3_FOUND)
    # Prefer a pybind11 already installed in the active environment (see
    # requirements.txt). Point CMake at a venv with
    #   -DPython3_EXECUTABLE=$PWD/.venv/bin/python
    # Only if none is found do we fall back to downloading it, so a prepared
    # venv keeps the build fully offline.
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND Python3_EXECUTABLE)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE _pybind11_rc)
      if(_pybind11_rc EQUAL 0 AND EXISTS "${_pybind11_dir}")
        find_package(pybind11 CONFIG QUIET PATHS "${_pybind11_dir}" NO_DEFAULT_PATH)
      endif()
    endif()

    if(pybind11_FOUND)
      message(STATUS "pybind11: using installed ${pybind11_VERSION}")
    else()
      message(STATUS "pybind11: not installed, fetching (needs network). "
                     "Install it into your venv to avoid this - see requirements.txt")
      include(FetchContent)
      FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE)
      FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_bd_csa python/bindings.cpp)
    if(BD_CSA_ENABLE_CUDA)
      target_link_libraries(_bd_csa PRIVATE bd_csa_cuda)
    else()
      target_link_libraries(_bd_csa PRIVATE bd_csa_core)
    endif()
    # Emit the module inside the package directory so `PYTHONPATH=python` works
    # without an install step.
    set_target_properties(_bd_csa PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/bd_csa)
  else()
    message(STATUS "Python3 development headers not found - skipping bindings")
  endif()
endif()

enable_testing()
add_subdirectory(tests)
