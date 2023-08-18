#ifndef UNCALIBRATED_VP_ESTIMATORS_UNCALIBRATED_VP_ESTIMATOR_BASE_H_
#define UNCALIBRATED_VP_ESTIMATORS_UNCALIBRATED_VP_ESTIMATOR_BASE_H_

#include "estimators/vp_estimator_base.h"
#include "refinement/ls_non_orthogonal.h"
#include "refinement/ls_orthogonal.h"

#include <Eigen/LU>
#include <Eigen/Geometry>
#include <iostream>
#include "ceres/ceres.h"

namespace uncalibrated_vp {

class Uncalibrated_VP_Estimator_Base: public VP_Estimator_Base {
public:
    using ResultType = std::pair<double, Eigen::Matrix3d>; // (focal length, R). In particular, the R is col major: [vp1, vp2, vp3]

    // The input data should be shifted with the principle point (0, 0)
    Uncalibrated_VP_Estimator_Base (const Eigen::Matrix4Xd& data, int ls_refinement = 0, int nms = 0, bool magsac_scoring = false): VP_Estimator_Base(data), ls_refinement_(ls_refinement), nms_(nms), magsac_scoring_(magsac_scoring) {}

    // minimal solver
    inline int min_sample_size() const { return 4; }
    virtual int MinimalSolver(const std::vector<int>& sample,
                              std::vector<ResultType>* res) const = 0;

    // measurements
    double EvaluateModelOnPoint(const ResultType& res, int i) const { return EvaluateOnePoint(res.second, res.first, i); }
    double Score(const Eigen::Matrix3d& R, const double focal_length, const double inlier_threshold) const;
    double Score(const Eigen::Matrix3d& R, const Eigen::Matrix3d& K, const double inlier_threshold) const;
    double MAGSACScore(const M3D& R, const M3D& K, const double inlier_threshold) const;
    double RANSACScore(const M3D& R, const M3D& K, const double inlier_threshold) const;
    double ScoreModel(const ResultType& res, const double inlier_threshold) const { return Score(res.second, res.first, inlier_threshold); }

    inline int non_minimal_sample_size() const { return 4; }
    int NonMinimalSolver(const std::vector<int>& sample, ResultType* res) const;
    void LeastSquares(const std::vector<int>& sample, ResultType* res) const {
        if (ls_refinement_ == 1)
            NonOrthogonalLeastSquares(sample, data_, &(res->second), res->first);
        else if (ls_refinement_ == 2)
            OrthogonalLeastSquares(sample, data_, &(res->second), res->first, true);
    }

protected:
    int ls_refinement_;
    int nms_;
    bool magsac_scoring_;
};

} // namespace uncalibrated_vp 

#endif

