#include "base/functions.h"
#include "base/pose.h"
#include "refinement/util.h"
#include "refinement/ls_orthogonal.h"
#include "refinement/cost_functions.h"

#include <ceres/ceres.h>

namespace uncalibrated_vp {

void OrthogonalLeastSquares(const std::vector<int>& sample,
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

    // Optimize 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 3;
    ceres::Solver::Summary summary;

    V4D qvec = RotationMatrixToQuaternion(R->transpose()); // qvec is row-major
    ceres::Problem problem;
    for (size_t i = 0; i < 3; i++) {
        // Define the optimization problem
        for (size_t j = 0; j < n_samples; j++) {
            if (closest_vp[j] != i)
                continue;

            // Mid point of the current line
            V2D midpoint = data[j].midpoint();
            double weight = data[j].length() / sum_length;
            ceres::LossFunction* vp_loss = new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5), weight, ceres::TAKE_OWNERSHIP);
            ceres::CostFunction* vp_cost_function = OrthogonalVpCostFunctor::Create(
                data[j].start[0], data[j].start[1], midpoint[0], midpoint[1], i);
            problem.AddResidualBlock(vp_cost_function, vp_loss, qvec.data(), &f);
        }
    }

    // parameterize
    if (problem.HasParameterBlock(qvec.data())) {
        ceres::LocalParameterization* quaternion_parameterization = 
            new ceres::QuaternionParameterization;
        SetQuaternionManifold(&problem, qvec.data());
    }
    if (problem.HasParameterBlock(&f) && (!optimize_f)) {
        problem.SetParameterBlockConstant(&f);
    }

    // Solve the optimization problem and update the VP
    ceres::Solve(options, &problem, &summary);
    *R = QuaternionToRotationMatrix(qvec).transpose();
}

} // namespace uncalibrated_vp 

