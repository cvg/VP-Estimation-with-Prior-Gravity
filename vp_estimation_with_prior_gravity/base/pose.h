#ifndef UNCALIBRATED_BASE_POSE_H_
#define UNCALIBRATED_BASE_POSE_H_

#include <Eigen/Core>
#include <limits>

namespace uncalibrated_vp {

/////////////////////////////////////////////
// From COLMAP
// [LINK] https://github.com/colmap/colmap/blob/dev/src/base/pose.h
/////////////////////////////////////////////
// Convert 3D rotation matrix to Quaternion representation.
//
// @param rot_mat        3x3 rotation matrix.
//
// @return               Unit Quaternion rotation coefficients (w, x, y, z).
Eigen::Vector4d RotationMatrixToQuaternion(const Eigen::Matrix3d& rot_mat);

// Convert Quaternion representation to 3D rotation matrix.
//
// @param qvec           Unit Quaternion rotation coefficients (w, x, y, z).
//
// @return               3x3 rotation matrix.
Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& qvec);

// Normalize Quaternion vector.
//
// @param qvec          Quaternion rotation coefficients (w, x, y, z).
//
// @return              Unit Quaternion rotation coefficients (w, x, y, z).
Eigen::Vector4d NormalizeQuaternion(const Eigen::Vector4d& qvec);

} // namespace uncalibrated_vp 

#endif

