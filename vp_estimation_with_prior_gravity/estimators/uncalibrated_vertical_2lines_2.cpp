#include "estimators/uncalibrated_vertical_2lines_2.h"
#include "solvers/solver_vp_2_vertical.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>

namespace uncalibrated_vp {

int Uncalibrated_Vertical_2lines_2::MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const {
    // the two lines share a vp
    if (sample.size() < 2u) return 0;

    std::vector<V3D> ls;
    ls.push_back(data_[sample[0]].coords());
    ls.push_back(data_[sample[1]].coords());

    std::vector<double> fs(1);
    std::vector<M3D> Rs(1);
    int num_sols = uncalibrated_vp_2_solver(v_, ls.data(), Rs.data(), fs.data());
    if (num_sols == 0)
        return 0;
    res->resize(1);
    (*res)[0] = std::make_pair(fs[0], Rs[0]);
    return 1;
}

} // namespace uncalibrated_vp 

