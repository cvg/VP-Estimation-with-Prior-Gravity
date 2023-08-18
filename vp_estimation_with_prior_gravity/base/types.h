#ifndef UNCALIBRATED_BASE_TYPES_H_
#define UNCALIBRATED_BASE_TYPES_H_

#include <Eigen/Core>
#include <limits>

namespace uncalibrated_vp {

const double EPS = std::numeric_limits<double>::min();
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;
using M3D = Eigen::Matrix3d;

inline V3D homogeneous(const V2D& v2d) { return V3D(v2d(0), v2d(1), 1.0); }
inline V2D dehomogeneous(const V3D& v3d) { return V2D(v3d(0) / (v3d(2) + EPS), v3d(1) / (v3d(2) + EPS)); }

} // namespace uncalibrated_vp 

#endif

