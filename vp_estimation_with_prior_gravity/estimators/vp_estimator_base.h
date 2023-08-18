#ifndef UNCALIBRATED_VP_ESTIMATORS_VP_ESTIMATOR_BASE_H_
#define UNCALIBRATED_VP_ESTIMATORS_VP_ESTIMATOR_BASE_H_

#include <vector>
#include "base/linebase.h"
#include "base/types.h"

namespace uncalibrated_vp {

class VP_Estimator_Base {
public:
    VP_Estimator_Base(const Eigen::Matrix4Xd& data); // 4 x n array, each col is [x1, y1, x2, y2]

    inline int num_data() const { return num_data_; }

    // Evaluates the rotation and focal length on the i-th data point.
    double EvaluateOnePoint(const Eigen::Matrix3d& R, const double focal_length, int i) const; // assume that the coordinates are shifted with (0, 0) principle point
    double EvaluateOnePoint(const Eigen::Matrix3d& R, const Eigen::Matrix3d& K, int i) const;

    double Score(const Eigen::Matrix3d& R, const double focal_length, const double inlier_threshold) const; // assume that the coordinates are shifted with (0, 0) principle point
    double Score(const Eigen::Matrix3d& R, const Eigen::Matrix3d& K, const double inlier_threshold) const;

protected:
    std::vector<limap::Line2d> data_;
    int num_data_;
};

} // namespace uncalibrated_vp 

#endif

