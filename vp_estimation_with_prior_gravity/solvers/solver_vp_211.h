#ifndef UNCALIBRATED_VP_SOLVERS_SOLVER_VP_211_H_
#define UNCALIBRATED_VP_SOLVERS_SOLVER_VP_211_H_

#include <Eigen/Core>
#include <Eigen/Dense>

// Petr Hruby
// July 2022

namespace uncalibrated_vp {

//compute a focal length + a rotation from 4 projections of lines in the 2-1-1 pattern
int solver_vp211(Eigen::Vector3d * ls, //an array of 4 lines;
				 double * fs, //an array of focal lengths consistent with the input
				 Eigen::Matrix3d * Rs); //an array of rotation matrices consistent with the input

} // namespace uncalibrated_vp 

#endif

