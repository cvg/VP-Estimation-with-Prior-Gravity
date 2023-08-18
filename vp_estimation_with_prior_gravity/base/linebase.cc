#include "base/linebase.h"
#include <cmath>
#include <Eigen/Geometry>

namespace limap {

Line2d::Line2d(V2D start_, V2D end_, double score_) {
    start = start_; 
    end = end_;
    score = score_;
}

Line2d::Line2d(const Eigen::Vector4d& seg) {
    start = V2D(seg(0), seg(1));
    end = V2D(seg(2), seg(3));
}

Line2d::Line2d(const Eigen::MatrixXd& seg) {
    start = V2D(seg(0, 0), seg(0, 1));
    end = V2D(seg(1, 0), seg(1, 1));
}

double Line2d::point_distance(const V2D& p) const {
    double projection = (p - start).dot(direction());
    if (projection < 0)
        return (p - start).norm();
    if (projection > length())
        return (p - end).norm();
    double dist_squared = (p - start).squaredNorm() - pow(projection, 2);
    double dist = sqrt(std::max(dist_squared, 0.0));
    return dist;
}

V3D Line2d::coords() const {
    V3D start_homo = V3D(start[0], start[1], 1.0);
    V3D end_homo = V3D(end[0], end[1], 1.0);
    return start_homo.cross(end_homo).normalized();
}

Eigen::MatrixXd Line2d::as_array() const {
    Eigen::MatrixXd arr(2, 2);
    arr(0, 0) = start[0]; arr(0, 1) = start[1];
    arr(1, 0) = end[0]; arr(1, 1) = end[1];
    return arr;
}

} // namespace limap

