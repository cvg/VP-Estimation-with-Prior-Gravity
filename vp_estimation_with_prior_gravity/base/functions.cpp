#include "base/functions.h"

#include <Eigen/Geometry>

namespace uncalibrated_vp {

double evaluate_one_point_2d(const V3D& vp_3d, const limap::Line2d& l, const M3D& K) {
    V3D vp_2d_homo = (K * vp_3d).normalized();
    V3D midpoint_homo = homogeneous(l.midpoint());
    V3D line_vp_midpoint = midpoint_homo.cross(vp_2d_homo);
    double dist = std::abs(homogeneous(l.start).dot(line_vp_midpoint));
    dist /= V2D(line_vp_midpoint[0], line_vp_midpoint[1]).norm();
    return dist;
}

V3D line_to_sphere(const limap::Line2d& line, const M3D& K) {
    M3D K_inv = K.inverse();
    V3D pt1 = K_inv * homogeneous(line.start);
    V3D pt2 = K_inv * homogeneous(line.end);
    return pt1.cross(pt2).normalized();
}

double evaluate_one_point_3d(const V3D& vp_3d, const limap::Line2d& line, const M3D& K) {
    V3D n = line_to_sphere(line, K);
    return n.dot(vp_3d);
}

void EvaluatePointOnModel(const M3D& R, const limap::Line2d& l, const M3D& K, double* ret) {
    // now we use 2d as default
    ret[0] = evaluate_one_point_2d(R.col(0), l, K);
    ret[1]= evaluate_one_point_2d(R.col(1), l, K);
    ret[2] = evaluate_one_point_2d(R.col(2), l, K);
}

} // namespace uncalibrated_vp 

