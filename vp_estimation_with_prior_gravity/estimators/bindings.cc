#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <RansacLib/sampling.h>
#include <RansacLib/ransac.h>
#include <RansacLib/hybrid_ransac.h>
#include <Eigen/Core>

#include "estimators/exhaustive_sampler.h"
#include "estimators/functions.h"
#include "estimators/vp_hybrid_ransac.h"
#include "estimators/functions_hybrid.h"
#include "refinement/vp2rot.h"

Eigen::Vector3d default_vertical(0., -1., 0.);

void bind_estimators(py::module& m) {
    using namespace uncalibrated_vp;

#define REGISTER_SAMPLER(SUFFIX, Sampler) \
    m.def(("run_calibrated_3lines" + SUFFIX).c_str(), &run_calibrated_3lines<Sampler<Calibrated_3lines>>); \
    m.def(("run_uncalibrated_4lines_220" + SUFFIX).c_str(), &run_uncalibrated_4lines<Uncalibrated_4lines_220, Sampler<Uncalibrated_4lines_220>>); \
    m.def(("run_uncalibrated_4lines_211" + SUFFIX).c_str(), &run_uncalibrated_4lines<Uncalibrated_4lines_211, Sampler<Uncalibrated_4lines_211>>); \
    m.def(("run_calibrated_vertical_1line" + SUFFIX).c_str(), &run_calibrated_vertical_1line<Sampler<Calibrated_Vertical_1line>>, \
          py::arg("lines"), py::arg("f"), py::arg("v") = default_vertical, \
          py::arg("options"), py::arg("ls_refinement") = 0, py::arg("nms") = 0); \
    m.def(("run_uncalibrated_vertical_2lines_2" + SUFFIX).c_str(), &run_uncalibrated_vertical_2lines<Uncalibrated_Vertical_2lines_2, Sampler<Uncalibrated_Vertical_2lines_2>>, \
          py::arg("lines"), py::arg("v") = default_vertical, \
          py::arg("options"), py::arg("ls_refinement") = 0, py::arg("nms") = 0, py::arg("magsac_scoring") = false); \
    m.def(("run_uncalibrated_vertical_2lines_11" + SUFFIX).c_str(), &run_uncalibrated_vertical_2lines<Uncalibrated_Vertical_2lines_11, Sampler<Uncalibrated_Vertical_2lines_11>>, \
          py::arg("lines"), py::arg("v") = default_vertical, \
          py::arg("options"), py::arg("ls_refinement") = 0, py::arg("nms") = 0, py::arg("magsac_scoring") = false); \
    m.def(("run_uncalibrated_vertical_2lines_11_v2" + SUFFIX).c_str(), &run_uncalibrated_vertical_2lines<Uncalibrated_Vertical_2lines_11_V2, Sampler<Uncalibrated_Vertical_2lines_11_V2>>, \
          py::arg("lines"), py::arg("v") = default_vertical, \
          py::arg("options"), py::arg("ls_refinement") = 0, py::arg("nms") = 0, py::arg("magsac_scoring") = false);  \
    m.def(("run_uncalibrated_vertical_2lines_011" + SUFFIX).c_str(), &run_uncalibrated_vertical_2lines<Uncalibrated_Vertical_2lines_011, Sampler<Uncalibrated_Vertical_2lines_011>>, \
          py::arg("lines"), py::arg("v") = default_vertical, \
          py::arg("options"), py::arg("ls_refinement") = 0, py::arg("nms") = 0, py::arg("magsac_scoring") = false);

#define SUFFIX std::string("")
    REGISTER_SAMPLER(SUFFIX, ransac_lib::UniformSampling);
#undef SUFFIX
#define SUFFIX std::string("_exs")
    REGISTER_SAMPLER(SUFFIX, ExhaustiveSampling);
#undef SUFFIX
#undef REGISTER_SAMPLER

    m.def("fit_vp_to_rot", &Vp2Rot);
}

void bind_hybrid_estimators(py::module& m) {
    using namespace uncalibrated_vp;
    m.def("run_hybrid_uncalibrated", &run_hybrid_uncalibrated);
}

void bind_ransaclib(py::module& m) {
    py::class_<ransac_lib::RansacStatistics>(m, "RansacStats")
        .def(py::init<>())
        .def_readwrite("num_iterations", &ransac_lib::RansacStatistics::num_iterations)
        .def_readwrite("best_num_inliers", &ransac_lib::RansacStatistics::best_num_inliers)
        .def_readwrite("best_model_score", &ransac_lib::RansacStatistics::best_model_score)
        .def_readwrite("inlier_ratio", &ransac_lib::RansacStatistics::inlier_ratio)
        .def_readwrite("inlier_indices", &ransac_lib::RansacStatistics::inlier_indices)
        .def_readwrite("number_lo_iterations", &ransac_lib::RansacStatistics::number_lo_iterations);
    
    py::class_<ransac_lib::RansacOptions>(m, "RansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::RansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::RansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::RansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::RansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::RansacOptions::random_seed_);

    py::class_<ransac_lib::LORansacOptions>(m, "LORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::LORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::LORansacOptions::max_num_iterations_)
        .def_readwrite("success_probability_", &ransac_lib::LORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_threshold_", &ransac_lib::LORansacOptions::squared_inlier_threshold_)
        .def_readwrite("random_seed_", &ransac_lib::LORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ransac_lib::LORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ransac_lib::LORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ransac_lib::LORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ransac_lib::LORansacOptions::min_sample_multiplicator_)
        .def_readwrite("non_min_sample_multiplier_", &ransac_lib::LORansacOptions::non_min_sample_multiplier_)
        .def_readwrite("lo_starting_iterations_", &ransac_lib::LORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ransac_lib::LORansacOptions::final_least_squares_);

    // hybrid ransac
    py::class_<ransac_lib::HybridRansacStatistics>(m, "HybridRansacStatistics")
        .def(py::init<>())
        .def_readwrite("num_iterations_total", &ransac_lib::HybridRansacStatistics::num_iterations_total)
        .def_readwrite("num_iterations_per_solver", &ransac_lib::HybridRansacStatistics::num_iterations_per_solver)
        .def_readwrite("best_num_inliers", &ransac_lib::HybridRansacStatistics::best_num_inliers)
        .def_readwrite("best_solver_type", &ransac_lib::HybridRansacStatistics::best_solver_type)
        .def_readwrite("best_model_score", &ransac_lib::HybridRansacStatistics::best_model_score)
        .def_readwrite("inlier_ratios", &ransac_lib::HybridRansacStatistics::inlier_ratios)
        .def_readwrite("inlier_indices", &ransac_lib::HybridRansacStatistics::inlier_indices)
        .def_readwrite("number_lo_iterations", &ransac_lib::HybridRansacStatistics::number_lo_iterations);

    py::class_<ransac_lib::HybridLORansacOptions>(m, "HybridLORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ransac_lib::HybridLORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ransac_lib::HybridLORansacOptions::max_num_iterations_)
        .def_readwrite("max_num_iterations_per_solver_", &ransac_lib::HybridLORansacOptions::max_num_iterations_per_solver_)
        .def_readwrite("success_probability_", &ransac_lib::HybridLORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_thresholds_", &ransac_lib::HybridLORansacOptions::squared_inlier_thresholds_)
        .def_readwrite("data_type_weights_", &ransac_lib::HybridLORansacOptions::data_type_weights_)
        .def_readwrite("random_seed_", &ransac_lib::HybridLORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ransac_lib::HybridLORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ransac_lib::HybridLORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ransac_lib::HybridLORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ransac_lib::HybridLORansacOptions::min_sample_multiplicator_)
        .def_readwrite("lo_starting_iterations_", &ransac_lib::HybridLORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ransac_lib::HybridLORansacOptions::final_least_squares_);
    
    using namespace uncalibrated_vp;
    py::class_<ExtendedHybridLORansacOptions>(m, "ExtendedHybridLORansacOptions")
        .def(py::init<>())
        .def_readwrite("min_num_iterations_", &ExtendedHybridLORansacOptions::min_num_iterations_)
        .def_readwrite("max_num_iterations_", &ExtendedHybridLORansacOptions::max_num_iterations_)
        .def_readwrite("max_num_iterations_per_solver_", &ExtendedHybridLORansacOptions::max_num_iterations_per_solver_)
        .def_readwrite("success_probability_", &ExtendedHybridLORansacOptions::success_probability_)
        .def_readwrite("squared_inlier_thresholds_", &ExtendedHybridLORansacOptions::squared_inlier_thresholds_)
        .def_readwrite("data_type_weights_", &ExtendedHybridLORansacOptions::data_type_weights_)
        .def_readwrite("random_seed_", &ExtendedHybridLORansacOptions::random_seed_)
        .def_readwrite("num_lo_steps_", &ExtendedHybridLORansacOptions::num_lo_steps_)
        .def_readwrite("threshold_multiplier_", &ExtendedHybridLORansacOptions::threshold_multiplier_)
        .def_readwrite("num_lsq_iterations_", &ExtendedHybridLORansacOptions::num_lsq_iterations_)
        .def_readwrite("min_sample_multiplicator_", &ExtendedHybridLORansacOptions::min_sample_multiplicator_)
        .def_readwrite("non_min_sample_multiplier_", &ExtendedHybridLORansacOptions::non_min_sample_multiplier_)
        .def_readwrite("lo_starting_iterations_", &ExtendedHybridLORansacOptions::lo_starting_iterations_)
        .def_readwrite("final_least_squares_", &ExtendedHybridLORansacOptions::final_least_squares_);
}

PYBIND11_MODULE(vp_estimation_with_prior_gravity_estimators, m){
    m.doc() = "pybind11 for customized estimators for vanishing points";
    bind_estimators(m);
    bind_hybrid_estimators(m);
    bind_ransaclib(m);
}
