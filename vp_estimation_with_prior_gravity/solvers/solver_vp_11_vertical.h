// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
inline int uncalibrated_vp_11_solver(Eigen::Vector3d v,  //a calibrated vertical direction
							Eigen::Vector3d * ls, //array of projections of two mutually orthogonal horizontal lines
							Eigen::Matrix3d * Rs, //output array of rotation matrices consistent with the input
							double * f); //output array of focal lengths consistent with the input

//FUNCTIONS
inline int uncalibrated_vp_11_solver(Eigen::Vector3d v, Eigen::Vector3d * ls, Eigen::Matrix3d * Rs, double * fs)
{
	const Eigen::Vector3d l1 = ls[0];
	const Eigen::Vector3d l2 = ls[1];
	v = v/v.norm();
	
	//obtain the orthonormal basis of the orthogonal complement to v
	Eigen::Vector3d b0(1,1,1);
	Eigen::Vector3d b1 = v.cross(b0);
	b1 = b1/b1.norm();
	Eigen::Vector3d b2 = v.cross(b1);
	b2 = b2/b2.norm();

	//find the coefficients of the univariate polynomial equation
	const double cc1 = (l1(2)*b1(2)*l2(0)*b1(0) +l1(2)*b1(2)*l2(1)*b1(1) -l1(2)*b2(2)*l2(0)*b2(0) -l1(2)*b2(2)*l2(1)*b2(1) -l2(2)*b1(2)*l1(0)*b1(0) +l2(2)*b2(2)*l1(0)*b2(0) -l2(2)*b1(2)*l1(1)*b1(1) +l2(2)*b2(2)*l1(1)*b2(1));
	const double cc2 = (l1(2)*b1(2)*l2(0)*b2(0) + l1(2)*b1(2)*l2(1)*b2(1) - l2(2)*b2(2)*l1(0)*b1(0) - l2(2)*b2(2)*l1(1)*b1(1));
	const double cc3 = (-l1(2)*b2(2)*l2(0)*b1(0) -l1(2)*b2(2)*l2(1)*b1(1) +l2(2)*b1(2)*l1(0)*b2(0) +l2(2)*b1(2)*l1(1)*b2(1));
	const double c0 = cc2;
	const double c1 = 2*cc1;
	const double c2 = 4*cc3-2*cc2;
	const double c3 = -2*cc1;
	const double c4 = cc2;
	
	//solve the equation with companion matrix approach
	Eigen::Matrix4d MM = Eigen::Matrix4d::Zero();
	MM(1,0) = 1;
	MM(2,1) = 1;
	MM(3,2) = 1;
	MM(0,3) = -c0/c4;
	MM(1,3) = -c1/c4;
	MM(2,3) = -c2/c4;
	MM(3,3) = -c3/c4;
	const Eigen::Vector4cd evs = MM.eigenvalues();
	
	//extract the rotation matrices and focal length
	int num_sols = 0;	
	for(int i=0;i<4;++i)
	{
		//skip complex solutions
		if(evs(i).imag() > 1e-5 || evs(i).imag() < -1e-5) continue;
		
		//extract the calibrated vps
		const double t = evs(i).real();
		Eigen::Vector3d d1 = (1-t*t)*b1 - 2*t*b2;
		d1 = d1/d1.norm();
		Eigen::Vector3d d2 = 2*t*b1 + (1-t*t)*b2;
		d2 = d2/d2.norm();
		
		//compute the focal length
		const double f = -l1(2)*d1(2)/(l1(0)*d1(0)+l1(1)*d1(1));
		if(f < 0) continue;
		
		//construct the rotation matrix
		Eigen::Matrix3d R;
		R.col(0) = v;
		R.col(1) = d1;
		R.col(2) = d2;
		if(R.determinant() < 0)
			R.col(2) = -d2;
			
		//store the result
		Rs[num_sols] = R;
		fs[num_sols] = f;
		++num_sols;
	}
		
	return num_sols;
}
