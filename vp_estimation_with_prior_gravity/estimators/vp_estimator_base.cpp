#include "estimators/vp_estimator_base.h"
#include "base/functions.h"

#include <string>
#include <cmath>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>

namespace uncalibrated_vp {

VP_Estimator_Base::VP_Estimator_Base(const Eigen::Matrix4Xd& data) {
    num_data_ = data.cols();
    for (size_t i = 0; i < num_data_; ++i) {
        data_.push_back(limap::Line2d(Eigen::Vector4d(data.col(i))));
    }
}

double VP_Estimator_Base::EvaluateOnePoint(const M3D& R, const double focal_length, int i) const {
    // assume that the coordinate is shifted with a (0, 0) principle point
    M3D K = M3D::Identity();
    K(0, 0) = K(1, 1) = focal_length;
    return EvaluateOnePoint(R, K, i);
}

double VP_Estimator_Base::EvaluateOnePoint(const M3D& R, const M3D& K, int i) const {
    double dist1 = evaluate_one_point_2d(R.col(0), data_[i], K);
    double dist2 = evaluate_one_point_2d(R.col(1), data_[i], K);
    double dist3 = evaluate_one_point_2d(R.col(2), data_[i], K);
    return std::min(dist1, std::min(dist2, dist3));
}

double VP_Estimator_Base::Score(const M3D& R, const double focal_length, const double inlier_threshold) const {
    // assume that the coordinate is shifted with a (0, 0) principle point
    M3D K = M3D::Identity();
    K(0, 0) = K(1, 1) = focal_length;
    return Score(R, K, inlier_threshold);
}

double VP_Estimator_Base::Score(const M3D& R, const M3D& K, const double inlier_threshold) const {
    double score = 0;
    for (size_t i = 0; i < num_data(); ++i) {
        double dist = EvaluateOnePoint(R, K, i);
        double lineLength =  1.0 / (data_[i].end -  data_[i].start).norm();
        double s = std::min(dist, inlier_threshold);
        score += lineLength * s;
    }
    return score;
}

} // namespace uncalibrated_vp 

