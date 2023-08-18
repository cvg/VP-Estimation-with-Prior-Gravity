#ifndef UNCALIBRATED_VP_REFINEMENT_COST_FUNCTIONS_H_
#define UNCALIBRATED_VP_REFINEMENT_COST_FUNCTIONS_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace uncalibrated_vp {

struct VpCostFunctor {
  explicit VpCostFunctor(
      double x1, double y1, double dcx, double dcy)
      : x1_(x1), y1_(y1), dcx_(dcx), dcy_(dcy) {}

  template <typename T>
  bool operator()(const T* vp, const T* f, T* residuals) const {
    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy_ * vp[2] / f[0] - vp[1];
    T l2 = vp[0] - dcx_ * vp[2] / f[0];
    T l3 = dcx_ * vp[1] - dcy_ * vp[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1_ * l1 + y1_ * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static ceres::CostFunction* Create(double x1, double y1, double dcx, double dcy) {
    return new ceres::AutoDiffCostFunction<VpCostFunctor, 1, 3, 1>(
        new VpCostFunctor(x1, y1, dcx, dcy));
  }

 private:
  const double x1_;
  const double y1_;
  const double dcx_;
  const double dcy_;
};

struct OrthogonalVpCostFunctor {
  explicit OrthogonalVpCostFunctor(
      double x1, double y1, double dcx, double dcy, int index)
      : x1_(x1), y1_(y1), dcx_(dcx), dcy_(dcy), index_(index) {}

  template <typename T>
  bool operator()(const T* qvec, const T* f, T* residuals) const {
    T R[3 * 3];
    ceres::QuaternionToRotation(qvec, R);
    T vp[3];
    for (size_t i = 0; i < 3; ++i) {
        vp[i] = R[3 * index_ + i];
    }

    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy_ * vp[2] / f[0] - vp[1];
    T l2 = vp[0] - dcx_ * vp[2] / f[0];
    T l3 = dcx_ * vp[1] - dcy_ * vp[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1_ * l1 + y1_ * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static ceres::CostFunction* Create(double x1, double y1, double dcx, double dcy, int index) {
    return new ceres::AutoDiffCostFunction<OrthogonalVpCostFunctor, 1, 4, 1>(
        new OrthogonalVpCostFunctor(x1, y1, dcx, dcy, index));
  }

 private:
  const double x1_;
  const double y1_;
  const double dcx_;
  const double dcy_;
  const int index_;
};

struct OrthogonalVerticalVpCostFunctor {
  explicit OrthogonalVerticalVpCostFunctor(
      const std::vector<V3D>& bases, double x1, double y1, double dcx, double dcy, int index)
      : bases_(bases), x1_(x1), y1_(y1), dcx_(dcx), dcy_(dcy), index_(index) {}

  template <typename T>
  bool operator()(const T* rvec, const T* f, T* residuals) const {
    T rvec_norm = ceres::sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1]);
    T vp[3];
    if (index_ == 0) {
        for (size_t i = 0; i < 3; ++i) {
            vp[i] = T(bases_[0][i]);
        }
    }
    else if (index_ == 1) {
        for (size_t i = 0; i < 3; ++i) {
            vp[i] = T(bases_[1][i] * rvec[0] + bases_[2][i] * rvec[1]) / rvec_norm;
        }
    }
    else { // index_ == 2
        for (size_t i = 0; i < 3; ++i) {
            vp[i] = T(bases_[1][i] * (-rvec[1]) + bases_[2][i] * rvec[0]) / rvec_norm;
        }
    }

    // Line passing through the VP and the center of the line segment
    // l = cross(dc, vp)
    T l1 = dcy_ * vp[2] / f[0] - vp[1];
    T l2 = vp[0] - dcx_ * vp[2] / f[0];
    T l3 = dcx_ * vp[1] - dcy_ * vp[0];

    // Residual = max orthogonal distance of the two endpoints to l
    residuals[0] = (ceres::abs(x1_ * l1 + y1_ * l2 + l3)
                    / ceres::sqrt(l1 * l1 + l2 * l2));

    return true;
  }

  static ceres::CostFunction* Create(std::vector<V3D>& bases, double x1, double y1, double dcx, double dcy, int index) {
    return new ceres::AutoDiffCostFunction<OrthogonalVerticalVpCostFunctor, 1, 2, 1>(
        new OrthogonalVerticalVpCostFunctor(bases, x1, y1, dcx, dcy, index));
  }

 private:
  const std::vector<V3D> bases_;
  const double x1_;
  const double y1_;
  const double dcx_;
  const double dcy_;
  const int index_;
};

struct Vp2RotCostFunctor {
  explicit Vp2RotCostFunctor(
      const V3D& vp, int index): vp_(vp), index_(index) {}

  template <typename T>
  bool operator()(const T* qvec, T* residuals) const {
    T R[3 * 3];
    ceres::QuaternionToRotation(qvec, R);
    residuals[0] = (R[3 * index_] * vp_[0]
                    + R[3 * index_ + 1] * vp_[1]
                    + R[3 * index_ + 2] * vp_[2]);
    return true;
  }

  static ceres::CostFunction* Create(const V3D& vp, int index) {
    return new ceres::AutoDiffCostFunction<Vp2RotCostFunctor, 1, 4>(
        new Vp2RotCostFunctor(vp, index));
  }

 private:
  const V3D vp_;
  int index_;
};

} // namespace uncalibrated_vp 

#endif

