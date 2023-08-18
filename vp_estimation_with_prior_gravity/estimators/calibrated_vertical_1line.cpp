#include "estimators/calibrated_vertical_1line.h"

#include "base/functions.h"
#include "solvers/solver_vp_calibrated_vertical.h"
#include "solvers/NMS_calibrated.h"

#include <Eigen/Geometry>

namespace uncalibrated_vp {

int Calibrated_Vertical_1line::MinimalSolver(const std::vector<int>& sample, 
        std::vector<Eigen::Matrix3d>* res) const {
    if (sample.size() < 1u) return 0;

    std::vector<M3D> Rs(1);
    int num_sols = calibrated_vertical_solver(v_, data_[sample[0]].coords(), f_, Rs.data());
    if (num_sols == 0)
        return 0;

    res->resize(1);
    (*res)[0] = Rs[0];
    return 1;
}

int Calibrated_Vertical_1line::NonMinimalSolver(const std::vector<int>& sample, ResultType* res) const {
    
    if (nms_ == 0) return 0;
    if (sample.size() < 3u) return 0;

    if (nms_ == 3) 
    {
        if (ls_refinement_ != 3)
        {
            Eigen::Matrix3d K;
            K << f_, 0, 0,
                0, f_, 0,
                0, 0, 1;

            V3D n1 = line_to_sphere(data_[sample[0]], K);
            V3D n2 = line_to_sphere(data_[sample[1]], K);
            V3D n3 = line_to_sphere(data_[sample[2]], K);

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
            (*res).col(0) = vp1;
            (*res).col(1) = vp2;
            (*res).col(2) = vp3;
        }

        LeastSquares(sample, res);
        return 1;
    }

    M3D K = M3D::Identity();
    K(0, 0) = f_;
    K(1, 1) = f_;
    size_t n_samples = sample.size();
    double dist_line_vp[3];
    int closest_vp;
    std::vector<std::vector<int>> inliers(3, std::vector<int>(0));
    for (int i = 0; i < n_samples; i++) {
        EvaluatePointOnModel(*res, data_[sample[i]], K, dist_line_vp);
        closest_vp = std::min_element(dist_line_vp, dist_line_vp + 3) - dist_line_vp;
        inliers[closest_vp].push_back(i);
    }

    // Check whether we have enough inliers for each VP
    if ((inliers[0].size() < 2) || (inliers[1].size() < 2)
        || (inliers[2].size() < 2)) return 0;

    // Get the line equation of the samples
    std::vector<Eigen::Vector3d> ls(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        ls[i] = data_[sample[i]].coords();

    // Run the non minimal solver
    if (nms_ == 1)
        NMS_calibrated_nonorthogonal(inliers[0], inliers[1], inliers[2], ls, *res);
    if (nms_ == 2)
        NMS_calibrated_linearized_iterative(inliers[0], inliers[1], inliers[2], ls, *res, 10);
    return 1;
}

} // namespace uncalibrated_vp 

