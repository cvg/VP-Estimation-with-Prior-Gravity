#ifndef UNCALIBRATED_VP_REFINEMENT_LS_ORTHOGONAL_H_
#define UNCALIBRATED_VP_REFINEMENT_LS_ORTHOGONAL_H_

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include "base/types.h"
#include "base/linebase.h"

namespace uncalibrated_vp {

void OrthogonalLeastSquares(
    const std::vector<int>& sample, const std::vector<limap::Line2d>& data, Eigen::Matrix3d* R, double f, bool optimize_f = true);

} // namespace uncalibrated_vp 

#endif
