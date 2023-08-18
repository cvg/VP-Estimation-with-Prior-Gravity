#include "estimators/hybrid_uncalibrated_vp_estimator.h"
#include "solvers/solver_vp_2_vertical.h"
#include "solvers/solver_vp_11_vertical_v2.h"
#include "solvers/solver_vp_011.h"
#include "solvers/solver_vp_211.h"
#include "solvers/solver_vp_220.h"

namespace uncalibrated_vp {

int Hybrid_Uncalibrated_VP_Estimator::MinimalSolver(const std::vector<std::vector<int>>& sample,
                                                    const int solver_idx, std::vector<ResultType>* res) const
{
    std::vector<V3D> ls;
    for (auto it = sample[0].begin(); it != sample[0].end(); ++it) {
        ls.push_back(data_[*it].coords());
    }
    std::vector<double> fs(32);
    std::vector<M3D> Rs(32);
    int num_sols = 0;
    if (solver_idx == 0)
        num_sols = uncalibrated_vp_2_solver(v_, ls.data(), Rs.data(), fs.data());
    else if (solver_idx == 1)
        num_sols = uncalibrated_vp_11_solver_v2(v_, ls.data(), Rs.data(), fs.data());
    else if (solver_idx == 2)
        num_sols = solver_vp_011(v_, ls.data(), Rs.data(), fs.data());
    else if (solver_idx == 3) 
        num_sols = solver_vp211(ls.data(), fs.data(), Rs.data());
    else if (solver_idx == 4)
        num_sols = solver_vp220(ls.data(), fs.data(), Rs.data());
    else
        throw std::runtime_error("Error! Not implemented!");

    res->resize(num_sols);
    for (size_t i = 0; i < num_sols; ++i) {
        ResultType res_i;
        if (solver_idx == 0 || solver_idx == 1 || solver_idx == 2)
            res_i = std::make_pair(fs[i], Rs[i]);
        else // (solver_idx == 3 || solver_idx == 4)
            res_i = std::make_pair(fs[i], Rs[i].transpose());
        (*res)[i] = res_i;
    }
    return num_sols;
}

void Hybrid_Uncalibrated_VP_Estimator::LeastSquares(const std::vector<std::vector<int>>& sample,
                                                    ResultType* res) const
{
    return Uncalibrated_Vertical_VP_Estimator_Base::LeastSquares(sample[0], res);
}

} // namespace uncalibrated_vp 

