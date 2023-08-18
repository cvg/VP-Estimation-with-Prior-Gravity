// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
inline int uncalibrated_vp_2_solver(Eigen::Vector3d v,  //a calibrated vertical direction
							Eigen::Vector3d * ls, //array of projections of two parallel horizontal lines
							Eigen::Matrix3d * Rs, //output array of rotation matrices consistent with the input
							double * f); //output array of focal lengths consistent with the input

//FUNCTIONS
inline int uncalibrated_vp_2_solver(Eigen::Vector3d v, Eigen::Vector3d * ls, Eigen::Matrix3d * Rs, double * fs)
{
	const Eigen::Vector3d l1 = ls[0];
	const Eigen::Vector3d l2 = ls[1];
	v = v/v.norm();
	const Eigen::Vector3d p1 = l1.cross(l2);
	const double f = -(v(0)*p1(0) + v(1)*p1(1))/(v(2)*p1(2));
	
	const Eigen::Vector3d l1c(f*l1(0), f*l1(1), l1(2));
	Eigen::Vector3d d1 = v.cross(l1c);
	d1 = d1/d1.norm();
	Eigen::Vector3d d2 = v.cross(d1);
	d2 = d2/d2.norm();
	
	Eigen::Matrix3d R;
	R.col(0) = v;
	R.col(1) = d1;
	R.col(2) = d2;
	if(R.determinant() < 0)
		R.col(2) = -d2;
	
	int num_sols = 0;
	Rs[num_sols] = R;
	fs[num_sols] = f;
	++num_sols;
		
	return num_sols;
}
