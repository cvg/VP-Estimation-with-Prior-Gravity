#ifndef UNCALIBRATED_VP_REFINEMENT_VP2ROT_H_
#define UNCALIBRATED_VP_REFINEMENT_VP2ROT_H_

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "base/functions.h"
#include "base/pose.h"
#include "refinement/cost_functions.h"


namespace uncalibrated_vp {

Eigen::Matrix3d Vp2Rot(const std::vector<V3D> &vp_triplet, Eigen::Matrix3d* R) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    // Define the problem
    V4D qvec = RotationMatrixToQuaternion(R->transpose()); // qvec is row-major
    ceres::Problem problem;
    for (size_t i = 0; i < 3; i++) {
        ceres::LossFunction* vp2rot_loss = new ceres::CauchyLoss(0.5);
        ceres::CostFunction* vp2rot_cost_function = Vp2RotCostFunctor::Create(
            vp_triplet[i], i);
        problem.AddResidualBlock(vp2rot_cost_function, vp2rot_loss, qvec.data());
    }

    // Parameterize
    if (problem.HasParameterBlock(qvec.data())) {
        ceres::LocalParameterization* quaternion_parameterization = 
            new ceres::QuaternionParameterization;
        problem.SetParameterization(qvec.data(), quaternion_parameterization);
    }

    // Solve the optimization problem and update the VP
    ceres::Solve(options, &problem, &summary);
    *R = QuaternionToRotationMatrix(qvec).transpose();

    return *R;
}

} // namespace uncalibrated_vp 

#endif
