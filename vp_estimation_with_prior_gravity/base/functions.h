#ifndef UNCALIBRATED_BASE_FUNCTIONS_H_
#define UNCALIBRATED_BASE_FUNCTIONS_H_

#include <Eigen/Core>
#include <limits>

#include "base/linebase.h"
#include "base/types.h"

namespace uncalibrated_vp {

double evaluate_one_point_2d(const Eigen::Vector3d& vp_3d, const limap::Line2d& l, const Eigen::Matrix3d& K);

Eigen::Vector3d line_to_sphere(const limap::Line2d& line, const Eigen::Matrix3d& K);

double evaluate_one_point_3d(const Eigen::Vector3d& vp_3d, const limap::Line2d& l, const Eigen::Matrix3d& K);

// now we use 2d as default
void EvaluatePointOnModel(const M3D& R, const limap::Line2d& l, const M3D& K, double* ret);

} // namespace uncalibrated_vp 

#endif

