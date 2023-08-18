#include "estimators/uncalibrated_vertical_vp_estimator_base.h"

#include "base/functions.h"
#include "solvers/NMS.h"
#include "gamma_values.cpp"

#include <Eigen/Geometry>

namespace uncalibrated_vp {

double Uncalibrated_Vertical_VP_Estimator_Base::RANSACScore(const M3D& R, const M3D& K, const double inlier_threshold) const {
    double score = 0;
    for (size_t i = 0; i < num_data(); ++i) {
        double dist = EvaluateOnePoint(R, K, i);
        double lineLength =  1.0 / (data_[i].end -  data_[i].start).norm();
        double s = std::min(dist, inlier_threshold);
        score += lineLength * s;
    }
    return score;
}

double Uncalibrated_Vertical_VP_Estimator_Base::MAGSACScore(const M3D& R, const M3D& K, const double inlier_threshold) const {
    // MAGSAC scoring, as in https://github.com/danini/magsac/blob/master/src/pymagsac/include/magsac.h
    // The degrees of freedom of the data from which the model is estimated.
	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
	constexpr size_t degrees_of_freedom = 4;
	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
	constexpr double k = 3.64;
	// A multiplier to convert residual values to sigmas
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	// Calculating k^2 / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double squared_k_per_2 = k * k / 2.0;
	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
	// Calculating (DoF + 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_plus_one_per_two = (degrees_of_freedom + 1.0) / 2.0;
	// TODO: check
	constexpr double C = 0.25;
	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_minus_one = std::pow(2.0, dof_minus_one_per_two);
	// Calculating 2^(DoF + 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_plus_one = std::pow(2.0, dof_plus_one_per_two);
	// Calculate the gamma value of k
	constexpr double gamma_value_of_k = 0.0036572608340910764;
	// Calculate the lower incomplete gamma value of k
	constexpr double lower_gamma_value_of_k = 1.3012265540498875;
	// Convert the maximum threshold to a sigma value
	const double maximum_sigma = threshold_to_sigma_multiplier * inlier_threshold;
	// Calculate the squared maximum sigma
	const double maximum_sigma_2 = maximum_sigma * maximum_sigma;
	// Calculate \sigma_{max}^2 / 2
	const double maximum_sigma_2_per_2 = maximum_sigma_2 / 2.0;
	// Calculate 2 * \sigma_{max}^2
	const double maximum_sigma_2_times_2 = maximum_sigma_2 * 2.0;
	// Calculate the loss implied by an outlier
	const double outlier_loss = maximum_sigma * two_ad_dof_minus_one  * lower_gamma_value_of_k;
	// Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	const double two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;
	// The loss which a point implies
	double loss = 0.0,
		// The total loss regarding the current model
		total_loss = 0.0;

	// Iterate through all points to calculate the implied loss
	for (size_t i = 0; i < num_data(); ++i)
	{
		// Calculate the residual of the current point
        const double residual = EvaluateOnePoint(R, K, i);

		// If the residual is smaller than the maximum threshold, consider it outlier
		// and add the loss implied to the total loss.
		if (inlier_threshold < residual)
			loss = outlier_loss;
		else // Otherwise, consider the point inlier, and calculate the implied loss
		{
			// Calculate the squared residual
			const double squared_residual = residual * residual;
			// Divide the residual by the 2 * \sigma^2
			const double squared_residual_per_sigma = squared_residual / maximum_sigma_2_times_2;
			// Get the position of the gamma value in the lookup table
			size_t x = round(precision_of_stored_incomplete_gammas * squared_residual_per_sigma);
			// If the sought gamma value is not stored in the lookup, return the closest element
			if (stored_incomplete_gamma_number < x)
				x = stored_incomplete_gamma_number;

			// Calculate the loss implied by the current point
			loss = maximum_sigma_2_per_2 * stored_lower_incomplete_gamma_values[x] +
				squared_residual / 4.0 * (stored_complete_gamma_values[x] -
					gamma_value_of_k);
			loss = loss * two_ad_dof_plus_one_per_maximum_sigma;
		}

		// Update the total loss
		total_loss += loss;
	}

	// Calculate the score of the model from the total loss
	return 1.0 / total_loss;
}

double Uncalibrated_Vertical_VP_Estimator_Base::Score(const M3D& R, const double focal_length, const double inlier_threshold) const {
    // assume that the coordinate is shifted with a (0, 0) principle point
    M3D K = M3D::Identity();
    K(0, 0) = K(1, 1) = focal_length;
    return Score(R, K, inlier_threshold);
}

double Uncalibrated_Vertical_VP_Estimator_Base::Score(const M3D& R, const M3D& K, const double inlier_threshold) const {
    if (magsac_scoring_)
        return MAGSACScore(R, K, inlier_threshold);
    else
        return RANSACScore(R, K, inlier_threshold);
}

int Uncalibrated_Vertical_VP_Estimator_Base::NonMinimalSolver(const std::vector<int>& sample, ResultType* res) const {
    // the first two lines correspond to the first vp, the third line corresponds to the second, and the fourth line corresponds to the third
    if (sample.size() < 4u) return 0;
    if (nms_ == 0) return 0;

    // Assign each sample to the closest VP
    M3D K = M3D::Identity();
    K(0, 0) = res->first;
    K(1, 1) = res->first;
    size_t n_samples = sample.size();
    double dist_line_vp[3];
    int closest_vp;
    std::vector<std::vector<int>> inliers(3, std::vector<int>(0));
    for (int i = 0; i < n_samples; i++) {
        EvaluatePointOnModel(res->second, data_[sample[i]], K, dist_line_vp);
        closest_vp = std::min_element(dist_line_vp, dist_line_vp + 3) - dist_line_vp;
        inliers[closest_vp].push_back(i);
    }

    // Check whether we have enough inliers for each VP
    if ((inliers[0].size() < 2) || (inliers[1].size() < 2)
        || (inliers[2].size() < 2)) return 0;

    // Get the line equation of the samples
    std::vector<Eigen::Vector3d> ls(n_samples);
    for (size_t i = 0; i < n_samples; i++)
        ls[i] = data_[sample[i]].coords();

    // Run the non minimal solver
    if (nms_ == 1)
        NMS_nonorthogonal(inliers[0], inliers[1], inliers[2], ls, res->second, res->first);
    if (nms_ == 2)
        NMS_linearized_iterative(inliers[0], inliers[1], inliers[2], ls, res->second, res->first, 10);
    
    if (nms_ == 3)
    {
        NMS_nonorthogonal(inliers[0], inliers[1], inliers[2], ls, res->second, res->first);
        LeastSquares(sample, res);
    }
    return 1;
}

} // namespace uncalibrated_vp 

