// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
inline int uncalibrated_vp_11_solver_v2(Eigen::Vector3d v,  //a calibrated vertical direction
							Eigen::Vector3d * ls, //array of projections of two mutually orthogonal horizontal lines
							Eigen::Matrix3d * Rs, //output array of rotation matrices consistent with the input
							double * f); //output array of focal lengths consistent with the input

//FUNCTIONS
inline int uncalibrated_vp_11_solver_v2(Eigen::Vector3d v, Eigen::Vector3d * ls, Eigen::Matrix3d * Rs, double * fs)
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
	
	const double A = l1(0)*b1(0) + l1(1)*b1(1);
	const double B = l1(2)*b1(2);
	const double C = l1(0)*b2(0) + l1(1)*b2(1);
	const double D = l1(2)*b2(2);
	const double E = l2(0)*b2(0) + l2(1)*b2(1);
	const double F = l2(2)*b2(2);
	const double G = l2(0)*b1(0) + l2(1)*b1(1);
	const double H = l2(2)*b1(2);
	
	
	//std::cout << Fgt*Fgt*(A*G+C*E) + Fgt*(A*H+B*G+C*F+D*E) + (B*H+D*F) << "\n";
	//std::cout << (A*G+C*E) << " " << (A*H+B*G+C*F+D*E) << " " << (B*H+D*F) << "\n";
	const double a = (A*G+C*E);
	const double b = (A*H+B*G+C*F+D*E);
	const double c = (B*H+D*F);
	const double d = b*b-4*a*c;
	const double sqrtd = std::sqrt(d);
	const double v1 = -b+sqrtd;
	const double v2 = -b-sqrtd;
	
	double f1;
	double f2;
	if(2*a > v1 || 2*a > v2)
	{
		f1 = v1/(2*a);
		f2 = v2/(2*a);
	}
	else
	{
		f1 = 2*c/v1;
		f2 = 2*c/v2;
	}
	//std::cout << Fgt << " " <<  << " " << v2/(2*a) << " " <<  << " " << 2*c/v2 << "\n";
	
	int num_sols = 0;
	if(f1 > 0)
	{
		const Eigen::Vector3d l1c(f1*l1(0), f1*l1(1), l1(2));
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
			
		Rs[num_sols] = R;
		fs[num_sols] = f1;
		++num_sols;
	}
	
	if(f2 > 0)
	{
		const Eigen::Vector3d l1c(f2*l1(0), f2*l1(1), l1(2));
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
			
		Rs[num_sols] = R;
		fs[num_sols] = f2;
		++num_sols;
	}
		
	return num_sols;
}
