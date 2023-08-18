#ifndef UNCALIBRATED_VP_ESTIMATORS_CALIBRATED_VERTICAL_1LINE_H_
#define UNCALIBRATED_VP_ESTIMATORS_CALIBRATED_VERTICAL_1LINE_H_

#include "estimators/vp_estimator_base.h"
#include "refinement/ls_non_orthogonal.h"
#include "refinement/ls_orthogonal.h"
#include "refinement/ls_orthogonal_vertical.h"
#include <Eigen/Core>

namespace uncalibrated_vp {

class Calibrated_Vertical_1line: public VP_Estimator_Base {
public:
    using ResultType = Eigen::Matrix3d; // col major: [vp1, vp2, vp3]
    // constructors
    Calibrated_Vertical_1line (
        const Eigen::Matrix4Xd& data, double f,
        int ls_refinement = 0, int nms = 0): VP_Estimator_Base(data), f_(f), ls_refinement_(ls_refinement), nms_(nms) {
        v_[0] = 0;
        v_[1] = -1;
        v_[2] = 0;
    }
    Calibrated_Vertical_1line (
        const Eigen::Matrix4Xd& data, double f, const Eigen::Vector3d& v,
        int ls_refinement = 0, int nms = 0): VP_Estimator_Base(data), f_(f), v_(v), ls_refinement_(ls_refinement), nms_(nms) {}

    // minimal solver
    inline int min_sample_size() const { return 1; }
    int MinimalSolver(const std::vector<int>& sample,
                      std::vector<ResultType>* res) const;

    // measurements
    double EvaluateModelOnPoint(const ResultType& res, int i) const { return EvaluateOnePoint(res, f_, i); }
    double ScoreModel(const ResultType& res, const double inlier_threshold) const { return Score(res, f_, inlier_threshold); }

    inline int non_minimal_sample_size() const { return 3; }
    int NonMinimalSolver(const std::vector<int>& sample, ResultType* res) const; 
    inline void LeastSquares(const std::vector<int>& sample, ResultType* res) const {
        if (ls_refinement_ == 1)
            NonOrthogonalLeastSquares(sample, data_, res, f_);
        else if (ls_refinement_ == 2)
            OrthogonalLeastSquares(sample, data_, res, f_, false);
        else if (ls_refinement_ == 3)
            OrthogonalVerticalLeastSquares(sample, v_, data_, res, f_, false);
    }
protected:
    Eigen::Vector3d v_;
    double f_;
    int ls_refinement_;
    int nms_;
};

} // namespace uncalibrated_vp 

#endif

