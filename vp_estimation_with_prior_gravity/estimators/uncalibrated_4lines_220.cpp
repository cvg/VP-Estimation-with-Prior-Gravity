#include "estimators/uncalibrated_4lines_220.h"
#include "solvers/solver_vp_220.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>

namespace uncalibrated_vp {

int Uncalibrated_4lines_220::MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const {
    // the first two lines share a vp, the last two lines share a vp
    if (sample.size() < 4u) return 0;

    std::vector<V3D> ls;
    ls.push_back(data_[sample[0]].coords());
    ls.push_back(data_[sample[1]].coords());
    ls.push_back(data_[sample[2]].coords());
    ls.push_back(data_[sample[3]].coords());

    std::vector<double> fs(4);
    std::vector<M3D> Rs(4);
    int num_sols = solver_vp220(ls.data(), fs.data(), Rs.data());
    if (num_sols == 0)
        return 0;
    res->resize(4);
    for (size_t i = 0; i < 4; ++i) {
        (*res)[i] = std::make_pair(fs[i], Rs[i].transpose());
    }
    return 4;
}

} // namespace uncalibrated_vp 

