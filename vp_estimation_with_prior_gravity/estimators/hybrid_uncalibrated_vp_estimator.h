#ifndef UNCALIBRATED_VP_ESTIMATORS_HYBRID_UNCALIBRATED_VP_ESTIMATOR_H_
#define UNCALIBRATED_VP_ESTIMATORS_HYBRID_UNCALIBRATED_VP_ESTIMATOR_H_

#include "estimators/uncalibrated_vertical_vp_estimator_base.h"

namespace uncalibrated_vp {

// Hybrid estimators for uncalibrated cases
// Solver 0 - 2lines 2
// Solver 1 - 2lines 11 v2
// Solver 2 - 2lines 011
// Solver 3 - 4lines 211
// Solver 4 - 4lines 220
class Hybrid_Uncalibrated_VP_Estimator: public Uncalibrated_Vertical_VP_Estimator_Base {
public:
    using ResultType = Uncalibrated_Vertical_VP_Estimator_Base::ResultType;

    Hybrid_Uncalibrated_VP_Estimator (
        const Eigen::Matrix4Xd& data, const Eigen::Vector3d& v,
        int ls_refinement = 0, int nms = 0, bool magsac_scoring = false): Uncalibrated_Vertical_VP_Estimator_Base(data, v, ls_refinement, nms, magsac_scoring) {}

    inline int num_minimal_solvers() const { return 5; }

    void min_sample_sizes(std::vector<std::vector<int>>* min_sample_sizes) const {
        min_sample_sizes->resize(5);
        (*min_sample_sizes)[0] = std::vector<int>{2};
        (*min_sample_sizes)[1] = std::vector<int>{2};
        (*min_sample_sizes)[2] = std::vector<int>{2};
        (*min_sample_sizes)[3] = std::vector<int>{4};
        (*min_sample_sizes)[4] = std::vector<int>{4};
    }

    inline int num_data_types() const { return 1; }

    void num_data(std::vector<int>* num_data) const {
        num_data->resize(1);
        (*num_data)[0] = data_.size();
    }

    int num_data() const { return data_.size(); }

    void solver_probabilities(std::vector<double>* solver_probabilities) const {
        solver_probabilities->resize(5);
        for (int i = 0; i < 5; i++) {
            if (solver_flags_[i])
                (*solver_probabilities)[i] = 1.0;
            else
                (*solver_probabilities)[i] = 0.0;
        }
    }

    int MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const override { return 0; } ; 

    int MinimalSolver(const std::vector<std::vector<int>>& sample,
                      const int solver_idx, std::vector<ResultType>* res) const;

    double EvaluateModelOnPoint(const ResultType& res, int t, int i) const { return Uncalibrated_Vertical_VP_Estimator_Base::EvaluateModelOnPoint(res, i); };

    void LeastSquares(const std::vector<std::vector<int>>& sample, ResultType* res) const;

    void set_solver_flags(const std::vector<bool>& solver_flags) { solver_flags_ = solver_flags; }

protected:
    std::vector<bool> solver_flags_ = std::vector<bool>{true, true, true, true, true};
};

} // namespace uncalibrated_vp 

#endif

