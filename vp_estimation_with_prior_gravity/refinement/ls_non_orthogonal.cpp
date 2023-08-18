#include "base/functions.h"
#include "refinement/ls_non_orthogonal.h"
#include "refinement/cost_functions.h"

namespace uncalibrated_vp {

void NonOrthogonalLeastSquares(const std::vector<int>& sample,
                               const std::vector<limap::Line2d>& data,
                               Eigen::Matrix3d* R, double f) {
    M3D K = M3D::Identity();
    K(0, 0) = f;
    K(1, 1) = f;
    
    // For each line, find the closest VP
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

    // Optimize independently each VP with its own inliers using Ceres
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 3;
    ceres::Solver::Summary summary;

    for (size_t i = 0; i < 3; i++) {
        V3D vp = R->col(i);

        // Do not perform any refinement if there are less than 2 inlier lines
        if (num_inliers[i] < 2)
            return;

        // Define the optimization problem
        ceres::Problem problem;
        for (size_t j = 0; j < n_samples; j++) {
            if (closest_vp[j] != i)
                continue;

            // Mid point of the current line
            V2D midpoint = data[j].midpoint();
            double weight = data[j].length() / sum_length;
            ceres::LossFunction* vp_loss = new ceres::ScaledLoss(
                new ceres::CauchyLoss(0.5), weight, ceres::TAKE_OWNERSHIP);
            ceres::CostFunction* vp_cost_function = VpCostFunctor::Create(
                data[j].start[0], data[j].start[1], midpoint[0], midpoint[1]);
            problem.AddResidualBlock(vp_cost_function, vp_loss, vp.data(), &f);
        }
        if (problem.HasParameterBlock(vp.data())) {
            ceres::LocalParameterization* homo3d_parameterization = new ceres::HomogeneousVectorParameterization(3);
            problem.SetParameterization(vp.data(), homo3d_parameterization);
        }
        if (problem.HasParameterBlock(&f))
            problem.SetParameterBlockConstant(&f);

        // Solve the optimization problem and update the VP
        ceres::Solve(options, &problem, &summary);
        R->col(i) = V3D(vp);
    }
}

} // namespace uncalibrated_vp 

