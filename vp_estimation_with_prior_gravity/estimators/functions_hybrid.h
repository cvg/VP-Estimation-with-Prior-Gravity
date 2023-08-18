#ifndef UNCALIBRATED_VP_ESTIMATORS_FUNCTIONS_HYBRID_H_
#define UNCALIBRATED_VP_ESTIMATORS_FUNCTIONS_HYBRID_H_

#include <iostream>
#include <utility>
#include <Eigen/Core>
#include <RansacLib/hybrid_ransac.h>
#include "estimators/vp_hybrid_ransac.h"
#include "estimators/hybrid_uncalibrated_vp_estimator.h"

namespace uncalibrated_vp {

// Lines should be shifted with a (0, 0) principle point
std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::HybridRansacStatistics> run_hybrid_uncalibrated(
    const Eigen::Matrix4Xd& lines,
    const Eigen::Vector3d& v,
    const ExtendedHybridLORansacOptions& options,
    int ls_refinement = 0,
    int nms = 0,
    bool magsac_scoring = false,
    bool sprt = false,
    const std::vector<bool>& solver_flags = std::vector<bool>{true, true, true, true, true});

} // namespace uncalibrated_vp 

#endif

