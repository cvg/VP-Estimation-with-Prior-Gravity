#include "estimators/functions_hybrid.h"

namespace uncalibrated_vp {

std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::HybridRansacStatistics> run_hybrid_uncalibrated(
    const Eigen::Matrix4Xd& lines,
    const Eigen::Vector3d& v,
    const ExtendedHybridLORansacOptions& options_,
    int ls_refinement,
    int nms,
    bool magsac_scoring,
    bool sprt,
    const std::vector<bool>& solver_flags) 
{
    ExtendedHybridLORansacOptions options = options_;
    std::random_device rand_dev;
    options.random_seed_ = rand_dev();

    // ransac
    using ResultType = std::pair<double, Eigen::Matrix3d>; // focal length, R (col-major)
    Hybrid_Uncalibrated_VP_Estimator estimator(lines, v, ls_refinement, nms, magsac_scoring);
    estimator.set_solver_flags(solver_flags);
    VPHybridRansac<ResultType, std::vector<ResultType>, Hybrid_Uncalibrated_VP_Estimator> hybrid_lomsac;
    hybrid_lomsac.set_sprt(sprt, estimator, options);
    ResultType best_model;
    ransac_lib::HybridRansacStatistics ransac_stats;
    int num_ransac_inliers = hybrid_lomsac.EstimateModel(options, estimator, &best_model, &ransac_stats);
    return std::make_pair(best_model, ransac_stats);
}

} // namespace uncalibrated_vp 

