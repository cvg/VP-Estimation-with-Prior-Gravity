// 
// \author Petr Hruby
// \date July 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES

inline int solver_vp_011(Eigen::Vector3d v, //the vertical direction
				  Eigen::Vector3d * ls, //two lines, first is vertical, second is horizontal
				  Eigen::Matrix3d * R, //rotation matrix
				  double * f); //focal length

//FUNCTIONS
inline int solver_vp_011(Eigen::Vector3d v, Eigen::Vector3d * ls, Eigen::Matrix3d * R, double * f)
{
	//find the focal length
	v = v/v.norm();
	const double F = -ls[0](2)*v(2)/(ls[0](0)*v(0) + ls[0](1)*v(1));
	f[0] = F;
	
	//find the direction of the horizontal line
	const Eigen::Vector3d KTl2(F*ls[1](0), F*ls[1](1), ls[1](2));
	// const Eigen::Vector3d d2 = (v.cross(KTl2)).normalized();
	Eigen::Vector3d d2 = v.cross(KTl2);
	d2 = d2 / d2.norm();
	
	//find the direction orthogonal to both previous ones
	// const Eigen::Vector3d d3 = (v.cross(d2)).normalized();
	Eigen::Vector3d d3 = v.cross(d2);
	d3 = d3 / d3.norm();
	
	//compose the rotation matrix
	R->col(0) = v;
	R->col(1) = d2;
	R->col(2) = d3;
	if(R->determinant() < 0)
		R->col(2) = -d3;
		
	return 1;
}
