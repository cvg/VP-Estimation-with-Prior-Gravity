#include "solvers/solver_vp_220.h"

#include <Eigen/Core>
#include <Eigen/Dense>

// Petr Hruby
// July 2022

namespace uncalibrated_vp {

int solver_vp220(Eigen::Vector3d * ls, double * fs, Eigen::Matrix3d * Rs)
{
	Eigen::Vector3d l0 = ls[0];
	Eigen::Vector3d l1 = ls[1];
	Eigen::Vector3d l2 = ls[2];
	Eigen::Vector3d l3 = ls[3];
	
	//compute the 1st and 2nd uncalibrated vp
	Eigen::Vector3d vp1 = l0.cross(l1);
	Eigen::Vector3d vp2 = l2.cross(l3);

	//compute the focal length from the vps + the assumption that they are orthogonal
	double ff = -(vp1(0)*vp2(0) + vp1(1)*vp2(1))/(vp1(2)*vp2(2));
	
	int num_sols = 0;
	if (ff > 0)
	{
		double f = std::sqrt(ff);
		
		//compute the rectified vps
		Eigen::Vector3d r1 = Eigen::Vector3d(vp1(0)/f, vp1(1)/f, vp1(2));
		r1 = r1/r1.norm();
		Eigen::Vector3d r2 = Eigen::Vector3d(vp2(0)/f, vp2(1)/f, vp2(2));
		r2 = r2/r2.norm();
		Eigen::Vector3d r3 = r1.cross(r2);
		r3 = r3/r3.norm();
		
		//check all 4 possible rotations
		Eigen::Matrix3d R1;
		R1 << r1,r2,r3;
		if(R1.determinant() < 0)
			R1 << r1,r2,-r3;
			
		Eigen::Matrix3d R2;
		R2 << -r1,r2,r3;
		if(R2.determinant() < 0)
			R2 << -r1,r2,-r3;
			
		Eigen::Matrix3d R3;
		R3 << r1,-r2,r3;
		if(R3.determinant() < 0)
			R3 << r1,-r2,-r3;
			
		Eigen::Matrix3d R4;
		R4 << -r1,-r2,r3;
		if(R4.determinant() < 0)
			R4 << -r1,-r2,-r3;
			
		fs[num_sols] = f;
		Rs[num_sols] = R1;
		++num_sols;
		
		fs[num_sols] = f;
		Rs[num_sols] = R2;
		++num_sols;
		
		fs[num_sols] = f;
		Rs[num_sols] = R3;
		++num_sols;
		
		fs[num_sols] = f;
		Rs[num_sols] = R4;
		++num_sols;
	}
	return num_sols;
}

} // namespace uncalibrated_vp 

