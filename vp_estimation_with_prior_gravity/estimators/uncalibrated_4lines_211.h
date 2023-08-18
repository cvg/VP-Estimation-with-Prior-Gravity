#ifndef UNCALIBRATED_VP_ESTIMATORS_UNCALIBRATED_4LINES_211_H_
#define UNCALIBRATED_VP_ESTIMATORS_UNCALIBRATED_4LINES_211_H_

#include "estimators/uncalibrated_vp_estimator_base.h"

namespace uncalibrated_vp {

class Uncalibrated_4lines_211: public Uncalibrated_VP_Estimator_Base {
public:
    using ResultType = Uncalibrated_VP_Estimator_Base::ResultType;

    // The input data should be shifted with the principle point (0, 0)
    Uncalibrated_4lines_211 (const Eigen::Matrix4Xd& data, int ls_refinement, int nms, bool magsac_scoring): Uncalibrated_VP_Estimator_Base(data, ls_refinement, nms, magsac_scoring) {}

    int MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const;
};

} // namespace uncalibrated_vp 

#endif

