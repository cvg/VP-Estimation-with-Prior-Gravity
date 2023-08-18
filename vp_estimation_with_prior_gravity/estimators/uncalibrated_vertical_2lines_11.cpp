#include "estimators/uncalibrated_vertical_2lines_11.h"
#include "solvers/solver_vp_11_vertical.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>

namespace uncalibrated_vp {

int Uncalibrated_Vertical_2lines_11::MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const {
    // the two lines correspond to different VPs
    if (sample.size() < 2u) return 0;

    std::vector<V3D> ls;
    ls.push_back(data_[sample[0]].coords());
    ls.push_back(data_[sample[1]].coords());

    std::vector<double> fs(4);
    std::vector<M3D> Rs(4);
    int num_sols = uncalibrated_vp_11_solver(v_, ls.data(), Rs.data(), fs.data());
    if (num_sols == 0)
        return 0;
    res->resize(num_sols);
    for (size_t i = 0; i < num_sols; ++i) {
        (*res)[i] = std::make_pair(fs[i], Rs[i]);
    }
    return num_sols;
}

} // namespace uncalibrated_vp 

