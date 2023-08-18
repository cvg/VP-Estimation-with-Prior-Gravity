#include "estimators/calibrated_3lines.h"
#include "base/functions.h"

#include <Eigen/Geometry>

namespace uncalibrated_vp {

int Calibrated_3lines::MinimalSolver(const std::vector<int>& sample, 
        std::vector<Eigen::Matrix3d>* res) const {
    if (sample.size() < 3u) return 0;

    V3D n1 = line_to_sphere(data_[sample[0]], K_);
    V3D n2 = line_to_sphere(data_[sample[1]], K_);
    V3D n3 = line_to_sphere(data_[sample[2]], K_);

    // compute the first vp from the first 2 lines
    V3D vp1, vp2, vp3;
    vp1 = n1.cross(n2);
    if (vp1.norm() < 1e-3)
        return 0;
    vp1 = vp1.normalized();

    // compute the second vp from the third line
    vp2 = vp1.cross(n3);
    if (vp2.norm() < 1e-3)
        return 0;
    vp2 = vp2.normalized();

    // compute the third vp with cross product
    vp3 = vp1.cross(vp2).normalized();

    // construct col-major rotation matrix
    res->resize(1);
    (*res)[0].col(0) = vp1;
    (*res)[0].col(1) = vp2;
    (*res)[0].col(2) = vp3;
    return 1;
}

} // namespace uncalibrated_vp 

