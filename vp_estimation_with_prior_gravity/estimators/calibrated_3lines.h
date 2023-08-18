#ifndef UNCALIBRATED_VP_ESTIMATORS_CALIBRATED_3LINES_H_
#define UNCALIBRATED_VP_ESTIMATORS_CALIBRATED_3LINES_H_

#include "estimators/vp_estimator_base.h"

namespace uncalibrated_vp {

class Calibrated_3lines: public VP_Estimator_Base {
public:
    using ResultType = Eigen::Matrix3d; // col major: [vp1, vp2, vp3]
    Calibrated_3lines (const Eigen::Matrix4Xd& data, const Eigen::Matrix3d& K): VP_Estimator_Base(data) { K_ = K; }

    // minimal solver
    inline int min_sample_size() const { return 3; }
    int MinimalSolver(const std::vector<int>& sample,
                      std::vector<ResultType>* res) const;

    // measurements
    double EvaluateModelOnPoint(const ResultType& res, int i) const { return EvaluateOnePoint(res, K_, i); }
    double ScoreModel(const ResultType& res, const double inlier_threshold) const { return Score(res, K_, inlier_threshold); }

    // TODO: now we do not have any non-minimal solver
    inline int non_minimal_sample_size() const { return 6; }
    int NonMinimalSolver(const std::vector<int>& sample, ResultType* res) const { return 0; } 
    inline void LeastSquares(const std::vector<int>& sample, ResultType* res) const {
        NonMinimalSolver(sample, res);
    }
protected:
    Eigen::Matrix3d K_;
};

} // namespace uncalibrated_vp 

#endif

