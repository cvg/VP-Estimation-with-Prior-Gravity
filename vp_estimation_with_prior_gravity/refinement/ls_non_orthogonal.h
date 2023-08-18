#ifndef UNCALIBRATED_VP_REFINEMENT_LS_NON_ORTHOGONAL_H_
#define UNCALIBRATED_VP_REFINEMENT_LS_NON_ORTHOGONAL_H_

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "base/types.h"
#include "base/linebase.h"

namespace uncalibrated_vp {

void NonOrthogonalLeastSquares(
    const std::vector<int>& sample, const std::vector<limap::Line2d>& data,
    Eigen::Matrix3d* R, double f);

} // namespace uncalibrated_vp 

#endif
