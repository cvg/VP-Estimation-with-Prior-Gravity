#ifndef LIMAP_BASE_LINEBASE_H
#define LIMAP_BASE_LINEBASE_H

#include <Eigen/Core>

namespace limap {

using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;

class Line2d {
public:
    Line2d() {}
    Line2d(const Eigen::MatrixXd& seg2d); // 2x2 array [[x1, y1], [x2, y2]]
    Line2d(const Eigen::Vector4d& seg2d); // [x1, y1, x2, y2]
    Line2d(V2D start, V2D end, double score=-1.0);
    V2D start, end;
    double score = -1.0;

    double length() const {return (start - end).norm();}
    V2D midpoint() const {return 0.5 * (start + end);}
    V2D direction() const {return (end - start).normalized();}
    V2D perp_direction() const {V2D dir = direction(); return V2D(dir[1], -dir[0]); }
    V3D coords() const;
    double point_distance(const V2D& p) const;
    Eigen::MatrixXd as_array() const;
};

} // namespace limap

#endif

