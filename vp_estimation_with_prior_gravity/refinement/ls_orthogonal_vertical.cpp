#include "base/functions.h"
#include "base/pose.h"
#include "refinement/ls_orthogonal_vertical.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace uncalibrated_vp {

void OrthogonalVerticalLeastSquares(const std::vector<int>& sample,
                                    const Eigen::Vector3d& vertical,
                                    const std::vector<limap::Line2d>& data,
                                    Eigen::Matrix3d* R, double f, bool optimize_f) {
    // For each line, find the closest VP
    M3D K = M3D::Identity();
    K(0, 0) = K(1, 1) = f;
    size_t n_samples = sample.size();
    double dist_line_vp[3];
    std::vector<int> num_inliers(3, 0);
    std::vector<int> closest_vp(n_samples);
    for (int i = 0; i < n_samples; i++) {
        EvaluatePointOnModel(*R, data[i], K, dist_line_vp);
        closest_vp[i] = std::min_element(dist_line_vp, dist_line_vp + 3) - dist_line_vp;
        num_inliers[closest_vp[i]]++;
    }
    // sum length (to get the final residual in pixel unit)
    double sum_length = 0;
    for (size_t i = 0; i < n_samples; ++i) {
        sum_length += data[i].length();
    }

    // Do not perform any refinement if there are less than 2 lines per VP
    if ((num_inliers[0] < 2) || (num_inliers[1] < 2) || (num_inliers[2] < 2))
        return;

    // Optimize: the first column of R equals to vertical 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 3;
    ceres::Solver::Summary summary;

    // get basis
    std::vector<V3D> bases(3);
    bases[0] = vertical;
    if (vertical.dot(V3D(1., 0., 0.)) == 1) {
        bases[1] = V3D(0., 1., 0.);
        bases[2] = V3D(0., 0., 1.);
    }
    else {
        bases[1] = vertical.cross(V3D(1., 0., 0.));
        bases[1] /= bases[1].norm();
        bases[2] = vertical.cross(bases[1]);
        bases[2] /= bases[2].norm();
    }

    V2D rvec;
    rvec(0) = R->col(1).dot(bases[1]);
    rvec(1) = R->col(1).dot(bases[2]);

    ceres::Problem problem;
    for (size_t i = 0; i < 3; i++) {
        // The vertical direction is not parameterized
        if (i == 0)
            continue;

        // Define the optimization problem
        for (size_t j = 0; j < n_samples; j++) {
            if (closest_vp[j] != i)
                continue;

            // Mid point of the current line
            V2D midpoint = data[j].midpoint();
            double weight = data[j].length() / sum_length;
            ceres::LossFunction* vp_loss = new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5), weight, ceres::TAKE_OWNERSHIP);
            ceres::CostFunction* vp_cost_function = OrthogonalVerticalVpCostFunctor::Create(
                bases, data[j].start[0], data[j].start[1], midpoint[0], midpoint[1], i);
            problem.AddResidualBlock(vp_cost_function, vp_loss, rvec.data(), &f);
        }
    }

    // parameterize
    if (problem.HasParameterBlock(rvec.data())) {
        ceres::LocalParameterization* homo2d_parameterization = 
            new ceres::HomogeneousVectorParameterization(2);
        problem.SetParameterization(rvec.data(), homo2d_parameterization);
    }
    if (problem.HasParameterBlock(&f) && (!optimize_f)) {
        problem.SetParameterBlockConstant(&f);
    }

    // Solve the optimization problem and update the VP
    rvec /= rvec.norm();
    ceres::Solve(options, &problem, &summary);
    R->col(0) = vertical;
    R->col(1) = rvec(0) * bases[1] + rvec(1) * bases[2];
    R->col(2) = -rvec(1) * bases[1] + rvec(0) * bases[2];
    if (R->determinant() < 0)
        R->col(2) *= -1;
}

} // namespace uncalibrated_vp 

