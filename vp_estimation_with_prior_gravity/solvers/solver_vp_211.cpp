#include "solvers/solver_vp_211.h"

// Petr Hruby
// July 2022

namespace uncalibrated_vp {

int solver_vp211(Eigen::Vector3d * ls, double * fs, Eigen::Matrix3d * Rs) {
	Eigen::Vector3d l0 = ls[0];
	Eigen::Vector3d l1 = ls[1];
	Eigen::Vector3d l2 = ls[2];
	Eigen::Vector3d l3 = ls[3];
	
	//perform a multiplication of the lines to get better stability of the problems -> we want balanced coefficients to have well conditioned problems
	//compute the average ratio between the l(2)/l(0) and l(2)/l(1) -> this will be the estimate of f
	//double f0 = l0(2)*l0(2)/(l0(0)*l0(0)) + l0(2)*l0(2)/(l0(1)*l0(1)) + l1(2)*l1(2)/(l1(0)*l1(0)) + l1(2)*l1(2)/(l1(1)*l1(1)) + l2(2)*l2(2)/(l2(0)*l2(0)) + l2(2)*l2(2)/(l2(1)*l2(1)) + l3(2)*l3(2)/(l3(0)*l3(0)) + l3(2)/l3(1);
	//double f0 = l0(2)*l0(2)/(l0(0)*l0(0) + l0(1)*l0(1)) + l1(2)*l1(2)/(l1(0)*l1(0) + l1(1)*l1(1)) + l2(2)*l2(2)/(l2(0)*l2(0) + l2(1)*l2(1)) + l3(2)*l3(2)/(l3(0)*l3(0) + l3(1)*l3(1));
	//f0 = std::sqrt(f0)/8;
	double max2 = 0;
	double max01 = 0;
	for(int i=0;i<4;++i)
	{
		if(ls[i](2) > max2)
			max2 = ls[i](2);
		else if(-ls[i](2) > max2)
			max2 = -ls[i](2);
			
		if(ls[i](0) > max01)
			max01 = ls[i](0);
		else if(-ls[i](0) > max01)
			max01 = -ls[i](0);
			
		if(ls[i](1) > max01)
			max01 = ls[i](1);
		else if(-ls[i](1) > max01)
			max01 = -ls[i](1);
	}
	double f0 = max2/max01;
	//f0 = 1;
	Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity();
	K0(0,0) = f0;
	K0(1,1) = f0;
	l0 = K0*l0;
	l1 = K0*l1;
	l2 = K0*l2;
	l3 = K0*l3;

	l0 = l0/l0.norm();
	l1 = l1/l1.norm();
	l2 = l2/l2.norm();
	l3 = l3/l3.norm();
	
	//SOLVE for vps, f, rotation
	//compute the 1st vanishing point from the 1st two lines
	Eigen::Vector3d vp1 = l0.cross(l1);
	vp1 = vp1/vp1.norm();
	
	//compute the first basis vector
	Eigen::Vector3d b1 = Eigen::Vector3d(vp1(1), -vp1(0), 0);
	b1 = b1/b1.norm(); //the equations assume normalized basis and vp1
	
	//compute the parameters for the equations and fill them into an Eigen::Vector
	//used in 1st equation
	double A = (double)(l2.transpose()*b1); //parameter, independent from the variables
	double B = l2(0)*vp1(0)*vp1(2) + l2(1)*vp1(1)*vp1(2);
	double C = -l2(2)*(vp1(0)*vp1(0) + vp1(1)*vp1(1));
	//used in both equations
	double D = (vp1(0)*vp1(0)*vp1(2)*vp1(2) + vp1(1)*vp1(1)*vp1(2)*vp1(2));
	double E = (vp1(0)*vp1(0) + vp1(1)*vp1(1))*(vp1(0)*vp1(0) + vp1(1)*vp1(1));
	//used in 2nd equation
	double F = (double)(l3.transpose()*b1);
	double G = l3(0)*vp1(0)*vp1(2) + l3(1)*vp1(1)*vp1(2);
	double H = -l3(2)*(vp1(0)*vp1(0) + vp1(1)*vp1(1));
	
	//create and fill the parameter vector
	Eigen::VectorXd data = Eigen::VectorXd(8);
	data(0) = A;
	data(1) = B;
	data(2) = C;
	data(3) = D;
	data(4) = E;
	data(5) = F;
	data(6) = G;
	data(7) = H;

	//solve the quadratic system that stems from the equations	
	double a = D*F*A+B*G;
	double b = E*F*A+B*H+C*G;
	double c = C*H;
	double d = b*b-4*a*c;
	double ff1 = (-b+std::sqrt(d))/(2*a);
	double ff2 = (-b-std::sqrt(d))/(2*a);
	double f1 = std::sqrt(ff1);
	double f2 = std::sqrt(ff2);

	int num_sols = 0;	
	if(ff1 > 0)
	{
		//get the calibration matrix
		Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
		K(0,0) = f1;
		K(1,1) = f1;
	
		//get the second basis vector
		Eigen::Vector3d b2 = Eigen::Vector3d(f1*vp1(0)*vp1(2), f1*vp1(1)*vp1(2), -vp1(0)*vp1(0)-vp1(1)*vp1(1));
		b2 = b2/b2.norm();
		
		//get the first rectified vanishing point
		Eigen::Vector3d r1 = Eigen::Vector3d(vp1(0)/f1, vp1(1)/f1, vp1(2));
		r1 = r1/r1.norm();
		
		//get the coefficients for the other two rectified vanishing points
		double c1 = l2.transpose()*K*b1;
		double c2 = -l2.transpose()*K*b2;
		double c3 = l3.transpose()*K*b1;
		double c4 = l3.transpose()*K*b2;
		Eigen::Matrix2d M;
		M(0,0) = c1;
		M(0,1) = c2;
		M(1,0) = c4;
		M(1,1) = c3;
		Eigen::JacobiSVD<Eigen::Matrix2d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix2d V = svd.matrixV();
		Eigen::Vector2d coef = V.col(1);
		Eigen::Vector3d r2 = coef(0)*b1 - coef(1)*b2;
		r2 = r2/r2.norm();
		Eigen::Vector3d r3 = coef(1)*b1 + coef(0)*b2;
		r3 = r3/r3.norm();
		
		//we know the directions of the rotation vectors but there are still 4 possible orientations of them that have to be considered (other 4 lead to a reflection, these may be ignored)
		Eigen::Matrix3d R1;
		R1 << r1,r2,r3;
		if(R1.determinant() < 0)
			R1 << -r1,r2,r3;
			
		fs[num_sols] = f0*f1;
		Rs[num_sols] = R1;
		++num_sols;
			
		Eigen::Matrix3d R2;
		R2 << r1,-r2,r3;
		if(R2.determinant() < 0)
			R2 << -r1,-r2,r3;
			
		fs[num_sols] = f0*f1;
		Rs[num_sols] = R2;
		++num_sols;
			
		Eigen::Matrix3d R3;
		R3 << r1,r2,-r3;
		if(R3.determinant() < 0)
			R3 << -r1,r2,-r3;
		
		fs[num_sols] = f0*f1;
		Rs[num_sols] = R3;
		++num_sols;
		
		Eigen::Matrix3d R4;
		R4 << r1,-r2,-r3;
		if(R4.determinant() < 0)
			R4 << -r1,-r2,-r3;
			
		fs[num_sols] = f0*f1;
		Rs[num_sols] = R4;
		++num_sols;
	}
	
	if(ff2 > 0)
	{
		//get the calibration matrix
		Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
		K(0,0) = f2;
		K(1,1) = f2;
	
		//get the second basis vector
		Eigen::Vector3d b2 = Eigen::Vector3d(f2*vp1(0)*vp1(2), f2*vp1(1)*vp1(2), -vp1(0)*vp1(0)-vp1(1)*vp1(1));
		b2 = b2/b2.norm();
		
		//get the first rectified vanishing point
		Eigen::Vector3d r1 = Eigen::Vector3d(vp1(0)/f2, vp1(1)/f2, vp1(2));
		r1 = r1/r1.norm();
		
		//get the coefficients for the other two rectified vanishing points
		double c1 = l2.transpose()*K*b1;
		double c2 = -l2.transpose()*K*b2;
		double c3 = l3.transpose()*K*b1;
		double c4 = l3.transpose()*K*b2;
		Eigen::Matrix2d M;
		M(0,0) = c1;
		M(0,1) = c2;
		M(1,0) = c4;
		M(1,1) = c3;
		Eigen::JacobiSVD<Eigen::Matrix2d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix2d V = svd.matrixV();
		Eigen::Vector2d coef = V.col(1);
		Eigen::Vector3d r2 = coef(0)*b1 - coef(1)*b2;
		r2 = r2/r2.norm();
		Eigen::Vector3d r3 = coef(1)*b1 + coef(0)*b2;
		r3 = r3/r3.norm();
		
		//we know the directions of the rotation vectors but there are still 4 possible orientations of them
		Eigen::Matrix3d R1;
		R1 << r1,r2,r3;
		if(R1.determinant() < 0)
			R1 << -r1,r2,r3;
			
		fs[num_sols] = f0*f2;
		Rs[num_sols] = R1;
		++num_sols;
			
		Eigen::Matrix3d R2;
		R2 << r1,-r2,r3;
		if(R2.determinant() < 0)
			R2 << -r1,-r2,r3;
			
		fs[num_sols] = f0*f2;
		Rs[num_sols] = R2;
		++num_sols;
			
		Eigen::Matrix3d R3;
		R3 << r1,r2,-r3;
		if(R3.determinant() < 0)
			R3 << -r1,r2,-r3;
			
		fs[num_sols] = f0*f2;
		Rs[num_sols] = R3;
		++num_sols;
			
		Eigen::Matrix3d R4;
		R4 << r1,-r2,-r3;
		if(R4.determinant() < 0)
			R4 << -r1,-r2,-r3;
		
		fs[num_sols] = f0*f2;
		Rs[num_sols] = R4;
		++num_sols;
	}
	return num_sols;
}

} // namespace uncalibrated_vp 

