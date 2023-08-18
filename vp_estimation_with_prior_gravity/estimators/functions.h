#ifndef UNCALIBRATED_VP_ESTIMATORS_FUNCTIONS_H_
#define UNCALIBRATED_VP_ESTIMATORS_FUNCTIONS_H_

#include <iostream>
#include <utility>
#include <tuple>
#include <Eigen/Core>
#include <RansacLib/ransac.h>
#include "estimators/calibrated_3lines.h"
#include "estimators/uncalibrated_4lines_220.h"
#include "estimators/uncalibrated_4lines_211.h"
#include "estimators/calibrated_vertical_1line.h"
#include "estimators/uncalibrated_vertical_2lines_2.h"
#include "estimators/uncalibrated_vertical_2lines_11.h"
#include "estimators/uncalibrated_vertical_2lines_11_v2.h"
#include "estimators/uncalibrated_vertical_2lines_011.h"

namespace uncalibrated_vp {

// For all functions below
// input lines: 4 x n array, with each column [x1, y1, x2, y2]
template <typename Sampler>
std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_3lines(const Eigen::Matrix4Xd& lines, const Eigen::Matrix3d& K, const ransac_lib::LORansacOptions& options);

// Here the lines should be ideally shifted with a (0, 0) principle point
template <typename Estimator, typename Sampler>
std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_4lines(
    const Eigen::Matrix4Xd& lines,
    const ransac_lib::LORansacOptions& options,
    int ls_refinement = 0,
    int nms = 0,
    bool magsac_scoring = false);

// Lines should be shifted with a (0, 0) principle point
template <typename Sampler>
std::pair<Eigen::Matrix3d, ransac_lib::RansacStatistics> run_calibrated_vertical_1line(
    const Eigen::Matrix4Xd& lines,
    double f,
    const Eigen::Vector3d& v,
    const ransac_lib::LORansacOptions& options,
    int ls_refinement = 0,
    int nms = 0);

// Lines should be shifted with a (0, 0) principle point
template <typename Estimator, typename Sampler>
std::pair<std::pair<double, Eigen::Matrix3d>, ransac_lib::RansacStatistics> run_uncalibrated_vertical_2lines(
    const Eigen::Matrix4Xd& lines,
    const Eigen::Vector3d& v,
    const ransac_lib::LORansacOptions& options,
    int ls_refinement = 0,
    int nms = 0,
    bool magsac_scoring = false);

} // namespace uncalibrated_vp 

#endif

