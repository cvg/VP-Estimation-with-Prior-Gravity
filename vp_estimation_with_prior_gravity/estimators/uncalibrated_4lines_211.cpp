#include "estimators/uncalibrated_4lines_211.h"
#include "solvers/solver_vp_211.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>

namespace uncalibrated_vp {

int Uncalibrated_4lines_211::MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const {
    // the first two lines correspond to the first vp, the third line corresponds to the second, and the fourth line corresponds to the third
    if (sample.size() < 4u) return 0;

    std::vector<V3D> ls;
    ls.push_back(data_[sample[0]].coords());
    ls.push_back(data_[sample[1]].coords());
    ls.push_back(data_[sample[2]].coords());
    ls.push_back(data_[sample[3]].coords());

    std::vector<double> fs(32);
    std::vector<M3D> Rs(32);
    int num_sols = solver_vp211(ls.data(), fs.data(), Rs.data());
    res->resize(num_sols);
    for (size_t i = 0; i < num_sols; ++i) {
        (*res)[i] = std::make_pair(fs[i], Rs[i].transpose());
    }
    return num_sols;
}

} // namespace uncalibrated_vp 

