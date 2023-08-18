#include "estimators/functions.h"
#include "estimators/exhaustive_sampler.h"
#include "estimators/vp_ransac.h"

#include <RansacLib/sampling.h>

namespace uncalibrated_vp {

// For all functions below
// input lines: 4 x n array, with each column [x1, y1, x2, y2]
template <typename Sampler>
std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_3lines(const Eigen::Matrix4Xd& lines, const Eigen::Matrix3d& K, const ransac_lib::LORansacOptions& options_) {
    ransac_lib::LORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    // ransac
    using ResultType = Eigen::Matrix3d;
    Calibrated_3lines estimator(lines, K);
    VPRansac<ResultType, std::vector<ResultType>, Calibrated_3lines, Sampler> lomsac;
    ResultType best_model;
    ransac_lib::RansacStatistics ransac_stats;
    int num_ransac_inliers = lomsac.EstimateModel(options, estimator, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

template <typename Estimator, typename Sampler>
std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_4lines(
    const Eigen::Matrix4Xd& lines,
    const ransac_lib::LORansacOptions& options_,
    int ls_refinement,
    int nms,
    bool magsac_scoring) {
    ransac_lib::LORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    // ransac
    using ResultType = std::pair<double, Eigen::Matrix3d>; // focal length, R (col-major)
    Estimator estimator(lines, ls_refinement, nms, magsac_scoring);
    VPRansac<ResultType, std::vector<ResultType>, Estimator, Sampler> lomsac;
    ResultType best_model;
    ransac_lib::RansacStatistics ransac_stats;
    int num_ransac_inliers = lomsac.EstimateModel(options, estimator, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

template <typename Sampler>
std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_vertical_1line(
    const Eigen::Matrix4Xd& lines, double f,
    const Eigen::Vector3d& v, const ransac_lib::LORansacOptions& options_,
    int ls_refinement, int nms) {
    ransac_lib::LORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    // ransac
    using ResultType = Eigen::Matrix3d;
    Calibrated_Vertical_1line estimator(lines, f, v, ls_refinement, nms);
    VPRansac<ResultType, std::vector<ResultType>, Calibrated_Vertical_1line, Sampler> lomsac;
    ResultType best_model;
    ransac_lib::RansacStatistics ransac_stats;
    int num_ransac_inliers = lomsac.EstimateModel(options, estimator, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

template <typename Estimator, typename Sampler>
std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_vertical_2lines(
    const Eigen::Matrix4Xd& lines,
    const Eigen::Vector3d& v,
    const ransac_lib::LORansacOptions& options_,
    int ls_refinement,
    int nms,
    bool magsac_scoring) {
    ransac_lib::LORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    // ransac
    using ResultType = std::pair<double, Eigen::Matrix3d>; // focal length, R (col-major)
    Estimator estimator(lines, v, ls_refinement, nms, magsac_scoring);
    VPRansac<ResultType, std::vector<ResultType>, Estimator, Sampler> lomsac;
    ResultType best_model;
    ransac_lib::RansacStatistics ransac_stats;
    int num_ransac_inliers = lomsac.EstimateModel(options, estimator, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

#define REGISTER_UNCALIBRATED_4LINES(Solver, Sampler) \
template std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_4lines<Solver, Sampler<Solver>>(const Eigen::Matrix4Xd& lines, const ransac_lib::LORansacOptions& options, int ls_refinement, int nms, bool magsac_scoring);

#define REGISTER_UNCALIBRATED_VERTICAL_2LINES(Solver, Sampler) \
template std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_vertical_2lines<Solver, Sampler<Solver>>(const Eigen::Matrix4Xd& lines, const Eigen::Vector3d& v, const ransac_lib::LORansacOptions& options, int ls_refinement, int nms, bool magsac_scoring); 

#define REGISTER_SAMPLER(Sampler) \
template std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_3lines<Sampler<Calibrated_3lines>>(const Eigen::Matrix4Xd& lines, const Eigen::Matrix3d& K, const ransac_lib::LORansacOptions& options_); \
REGISTER_UNCALIBRATED_4LINES(Uncalibrated_4lines_220, Sampler) \
REGISTER_UNCALIBRATED_4LINES(Uncalibrated_4lines_211, Sampler) \
template std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_vertical_1line<Sampler<Calibrated_Vertical_1line>>(const Eigen::Matrix4Xd& lines, double f, const Eigen::Vector3d& v, const ransac_lib::LORansacOptions& options_, int ls_refinement, int nms); \
REGISTER_UNCALIBRATED_VERTICAL_2LINES(Uncalibrated_Vertical_2lines_2, Sampler) \
REGISTER_UNCALIBRATED_VERTICAL_2LINES(Uncalibrated_Vertical_2lines_11, Sampler) \
REGISTER_UNCALIBRATED_VERTICAL_2LINES(Uncalibrated_Vertical_2lines_11_V2, Sampler) \
REGISTER_UNCALIBRATED_VERTICAL_2LINES(Uncalibrated_Vertical_2lines_011, Sampler) \

REGISTER_SAMPLER(ransac_lib::UniformSampling);
REGISTER_SAMPLER(ExhaustiveSampling);

#undef REGISTER_SAMPLER
#undef REGISTER_UNCALIBRATED_4LINES
#undef REGISTER_UNCALIBRATED_VERTICAL_2LINES

}

// namespace uncalibrated_vp 

