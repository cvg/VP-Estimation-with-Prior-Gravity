#ifndef UNCALIBRATED_VP_SOLVERS_SOLVER_VP_220_H_
#define UNCALIBRATED_VP_SOLVERS_SOLVER_VP_220_H_

#include <Eigen/Core>
#include <Eigen/Dense>

// Petr Hruby
// July 2022

namespace uncalibrated_vp {

int solver_vp220(Eigen::Vector3d * ls, double * fs, Eigen::Matrix3d * Rs);

} // namespace uncalibrated_vp 

#endif

