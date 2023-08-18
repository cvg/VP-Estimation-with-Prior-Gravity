// #ifndef UNCALIBRATED_VP_SOLVERS_NMS_CALIBRATED_H_
// #define UNCALIBRATED_VP_SOLVERS_NMS_CALIBRATED_H_

// 
// \author Petr Hruby
// \date August 2022
// #include <vector>
#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
inline void NMS_calibrated_nonorthogonal(std::vector<int> &inliers_1, std::vector<int> &inliers_2, std::vector<int> &inliers_3, std::vector<Eigen::Vector3d> &ls, Eigen::Matrix3d &R);

inline void NMS_calibrated_linearized_iterative(std::vector<int> &inliers_1, std::vector<int> &inliers_2, std::vector<int> &inliers_3, std::vector<Eigen::Vector3d> &ls, Eigen::Matrix3d &R, int num_iters);

//FUNCTIONS
inline void NMS_calibrated_nonorthogonal(std::vector<int> &inliers_1, std::vector<int> &inliers_2, std::vector<int> &inliers_3, std::vector<Eigen::Vector3d> &ls, Eigen::Matrix3d &R)
{
	//fill the matrices with the inliers
	Eigen::MatrixXd A0(inliers_1.size(),3);
	for(int i=0;i<inliers_1.size();++i)
	{
		A0.block<1,3>(i,0) = ls[inliers_1[i]].transpose()/ls[inliers_1[i]].block<2,1>(0,0).norm();
	}
	Eigen::MatrixXd A1(inliers_2.size(),3);
	for(int i=0;i<inliers_2.size();++i)
	{
		A1.block<1,3>(i,0) = ls[inliers_2[i]].transpose()/ls[inliers_2[i]].block<2,1>(0,0).norm();
	}
	Eigen::MatrixXd A2(inliers_3.size(),3);
	for(int i=0;i<inliers_3.size();++i)
	{
		A2.block<1,3>(i,0) = ls[inliers_3[i]].transpose()/ls[inliers_3[i]].block<2,1>(0,0).norm();
	}
	
	//run SVD to find the best vanishing points A0*v0~0, ...
	Eigen::JacobiSVD<Eigen::MatrixXd> svd0(A0, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Vector3d v0 = svd0.matrixV().col(2);
	if(v0(2) < 0)
		v0 = -1*v0;
	
	Eigen::JacobiSVD<Eigen::MatrixXd> svd1(A1, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Vector3d v1 = svd1.matrixV().col(2);
	if(v1(2) < 0)
		v1 = -1*v1;
	
	Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A2, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Vector3d v2 = svd2.matrixV().col(2);
	if(v2(2) < 0)
		v2 = -1*v2;
	
	//compute the calibrated vps
	Eigen::Matrix3d KI = Eigen::Matrix3d::Identity();
	Eigen::Vector3d d0 = v0;
	d0=d0/d0.norm();
	Eigen::Vector3d d1 = v1;
	d1=d1/d1.norm();
	Eigen::Vector3d d2 = v2;
	d2=d2/d2.norm();
	
	//correct the calibrated vps to be orthogonal
	Eigen::Matrix3d D;
	D.col(0) = d0;
	D.col(1) = d1;
	D.col(2) = d2;
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	R = U*V.transpose();
}

inline void NMS_calibrated_linearized_iterative(std::vector<int> &inliers_1, std::vector<int> &inliers_2, std::vector<int> &inliers_3, std::vector<Eigen::Vector3d> &ls, Eigen::Matrix3d &R, int num_iters)
{
	//fill the matrices with the inliers
	Eigen::MatrixXd A0(inliers_1.size(),3);
	for(int i=0;i<inliers_1.size();++i)
	{
		A0.block<1,3>(i,0) = ls[inliers_1[i]].transpose()/ls[inliers_1[i]].block<2,1>(0,0).norm();
	}
	Eigen::MatrixXd A1(inliers_2.size(),3);
	for(int i=0;i<inliers_2.size();++i)
	{
		A1.block<1,3>(i,0) = ls[inliers_2[i]].transpose()/ls[inliers_2[i]].block<2,1>(0,0).norm();
	}
	Eigen::MatrixXd A2(inliers_3.size(),3);
	for(int i=0;i<inliers_3.size();++i)
	{
		A2.block<1,3>(i,0) = ls[inliers_3[i]].transpose()/ls[inliers_3[i]].block<2,1>(0,0).norm();
	}
	
	for(int i=0;i<num_iters;++i)
	{
		//build the matrices with the coefficients for the vp update (= transformation matrices)
		Eigen::Matrix<double,3,3> B0 = Eigen::Matrix<double,3,3>::Zero();
		Eigen::Vector3d C0;
		B0(0,1) = -R(2,0);
		B0(0,2) = R(1,0);
		C0(0) = R(0,0);
		
		B0(1,0) = R(2,0);
		B0(1,2) = -R(0,0);
		C0(1) = R(1,0);
		
		B0(2,0) = -R(1,0);
		B0(2,2) = R(0,0);
		C0(2) = R(2,0);
		
		Eigen::Matrix<double,3,3> B1 = Eigen::Matrix<double,3,3>::Zero();
		Eigen::Vector3d C1;
		B1(0,1) = -R(2,1);
		B1(0,2) = R(1,1);
		C1(0) = R(0,1);
		
		B1(1,0) = R(2,1);
		B1(1,2) = -R(0,1);
		C1(1) = R(1,1);
		
		B1(2,0) = -R(1,1);
		B1(2,2) = R(0,1);
		C1(2) = R(2,1);
		
		Eigen::Matrix<double,3,3> B2 = Eigen::Matrix<double,3,3>::Zero();
		Eigen::Vector3d C2;
		B2(0,1) = -R(2,2);
		B2(0,2) = R(1,2);
		C2(0) = R(0,2);
		
		B2(1,0) = R(2,2);
		B2(1,2) = -R(0,2);
		C2(1) = R(1,2);
		
		B2(2,0) = -R(1,2);
		B2(2,2) = R(0,2);
		C2(2) = R(2,2);
		
		//get the transformed points
		Eigen::MatrixXd AB0 = A0*B0;
		Eigen::MatrixXd AC0 = A0*C0;
		Eigen::MatrixXd AB1 = A1*B1;
		Eigen::MatrixXd AC1 = A1*C1;
		Eigen::MatrixXd AB2 = A2*B2;
		Eigen::MatrixXd AC2 = A2*C2;
		
		Eigen::MatrixXd AB(inliers_1.size()+inliers_2.size()+inliers_3.size(), 3);
		AB << AB0, AB1, AB2;
		Eigen::MatrixXd AC(inliers_1.size()+inliers_2.size()+inliers_3.size(), 1);
		AC << AC0, AC1, AC2;
		
		//solve for the best update
		Eigen::Vector3d dx = AB.colPivHouseholderQr().solve(-AC);
		
		//update the vps //TODO create both versions: update f and don't update f
		Eigen::Vector3d v0;
		v0(0) = R(0,0)+R(1,0)*dx(2)-R(2,0)*dx(1);
		v0(1) = R(1,0)-R(0,0)*dx(2)+R(2,0)*dx(0);
		v0(2) = R(2,0) + R(0,0)*dx(1) - R(1,0)*dx(0);
		
		Eigen::Vector3d v1;
		v1(0) = R(0,1)+R(1,1)*dx(2)-R(2,1)*dx(1);
		v1(1) = R(1,1)-R(0,1)*dx(2)+R(2,1)*dx(0);
		v1(2) = R(2,1) + R(0,1)*dx(1) - R(1,1)*dx(0);
		
		Eigen::Vector3d v2;
		v2(0) = R(0,2)+R(1,2)*dx(2)-R(2,2)*dx(1);
		v2(1) = R(1,2)-R(0,2)*dx(2)+R(2,2)*dx(0);
		v2(2) = R(2,2) + R(0,2)*dx(1) - R(1,2)*dx(0);
		
		//compute the calibrated vps
		Eigen::Vector3d d0 = v0;
		d0=d0/d0.norm();
		Eigen::Vector3d d1 = v1;
		d1=d1/d1.norm();
		Eigen::Vector3d d2 = v2;
		d2=d2/d2.norm();
		
		//correct the calibrated vps to be orthogonal
		Eigen::Matrix3d D;
		D.col(0) = d0;
		D.col(1) = d1;
		D.col(2) = d2;
		Eigen::Matrix3d DDT = D*D.transpose();
		if(DDT(0,2) < 1e-10 && DDT(0,2) > -1e-10)
		{
			R = D;
			continue;
		}
		//std::cerr << D*D.transpose() << "\n\n";
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(D, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		//update the rotation
		R = U*V.transpose();
	}
}

// #endif
