#ifndef UNCALIBRATED_VERTICAL_VP_ESTIMATORS_UNCALIBRATED_VERTICAL_2LINES_11_V2_H_
#define UNCALIBRATED_VERTICAL_VP_ESTIMATORS_UNCALIBRATED_VERTICAL_2LINES_11_V2_H_

#include "estimators/uncalibrated_vertical_vp_estimator_base.h"

namespace uncalibrated_vp {

class Uncalibrated_Vertical_2lines_11_V2: public Uncalibrated_Vertical_VP_Estimator_Base {
public:
    using ResultType = Uncalibrated_Vertical_VP_Estimator_Base::ResultType;

    // The input data should be shifted with the principle point (0, 0)
    Uncalibrated_Vertical_2lines_11_V2 (
        const Eigen::Matrix4Xd& data, const Eigen::Vector3d v,
        int ls_refinement, int nms, bool magsac_scoring): Uncalibrated_Vertical_VP_Estimator_Base(data, v, ls_refinement, nms, magsac_scoring) {}

    int MinimalSolver(const std::vector<int>& sample, std::vector<ResultType>* res) const;
};

} // namespace uncalibrated_vp 

#endif

