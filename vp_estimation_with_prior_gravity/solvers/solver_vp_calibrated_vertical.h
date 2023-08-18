// 
// \author Petr Hruby
// \date July 2022
// #include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

//PROTOTYPES
int calibrated_vertical_solver(Eigen::Vector3d v, //the vertical direction
							   Eigen::Vector3d l1, //projection of a horizontal line 
							   double f,  //focal length
							   Eigen::Matrix3d * R); //pointer to the resulting rotation matrix

//FUNCTIONS
int calibrated_vertical_solver(Eigen::Vector3d v, Eigen::Vector3d l1, double f, Eigen::Matrix3d * R)
{
	const Eigen::Vector3d l1c(f*l1(0), f*l1(1), l1(2));
	//get the direction d1 that is orthogonal to both the vertical direction d0 and to the line l1c
	v = v/v.norm();
	Eigen::Vector3d d1 = v.cross(l1c);
	d1 = d1/d1.norm();
	
	//get the direction d2 that is orthogonal to both 
	Eigen::Vector3d d2 = v.cross(d1);
	d2 = d2/d2.norm();
	
	//compose the rotation matrix
	R->col(0) = v;
	R->col(1) = d1;
	R->col(2) = d2;
	if(R->determinant() < 0)
		R->col(2) = -d2;
		
	return 1;
}
