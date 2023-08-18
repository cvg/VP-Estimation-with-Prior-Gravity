# cmake version to be used
cmake_minimum_required( VERSION 3.17 )

# project name
project(vp_estimation_with_prior_gravity)

################################################################################
# Include CMake dependencies
################################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version needs to be at least 9.1")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++14")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)

################################################################################
# Find packages
################################################################################
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

################################################################################
# Compiler specific configuration
################################################################################

if(OPENMP_ENABLED)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

################################################################################
# Add sources
################################################################################

set(UNCALIBRATED_VP_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(UNCALIBRATED_VP_EXTERNAL_LIBRARIES
  ${HDF5_C_LIBRARIES}
)

if(OPENMP_FOUND)
    list(APPEND UNCALIBRATED_VP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

set(UNCALIBRATED_VP_INTERNAL_LIBRARIES
    Eigen3::Eigen
    ceres
    HighFive
    pybind11::module
)

include_directories(
    third_party
    vp_estimation_with_prior_gravity
    ${UNCALIBRATED_VP_INCLUDE_DIRS}
)

add_subdirectory(third_party)
include_directories(${RANSACLIB_INCLUDE_DIRS})
add_subdirectory(vp_estimation_with_prior_gravity)
