#ifndef UNCALIBRATED_VP_ESTIMATORS_SPRT_PREEMPTIVE_VERIFICATION_H_
#define UNCALIBRATED_VP_ESTIMATORS_SPRT_PREEMPTIVE_VERIFICATION_H_

#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <chrono>
#include <limits>

#include <RansacLib/hybrid_ransac.h>

namespace uncalibrated_vp {

    using namespace ransac_lib;

    template <class Model, class ModelVector, class HybridSolver>
    class SPRTPreemptiveVerification
    {
    protected:
        class SPRTHistory {
        public:
            double epsilon, delta, A;
            // k is number of samples processed by test
            size_t k;
        };

        /*
        * The probability of a data point being consistent
        * with a ‘bad’ model is modeled as a probability of
        * a random event with Bernoulli distribution with parameter
        * δ : p(1|Hb) = δ.
        */

        /*
            * The probability p(1|Hg) = ε
            * that any randomly chosen data point is consistent with a ‘good’ model
            * is approximated by the fraction of inliers ε among the data
            * points
            */

            /*
            * The decision threshold A is the only parameter of the Adapted SPRT
            */
            // i
        size_t current_sprt_idx;

        double t_M, m_S, threshold, confidence;
        size_t points_size, sample_size, max_iterations, random_pool_idx;
        std::vector<SPRTHistory> sprt_histories;
        
        int number_rejected_models;
        int sum_fraction_data_points;
        size_t * points_random_pool;

        int max_hypothesis_test_before_sprt;

    public:
        double additional_model_probability;

        ~SPRTPreemptiveVerification() {
            delete[] points_random_pool;
        }

        // This function is called only once to calculate the exact model estimation time of 
        // the current model on the current machine. It is required for SPRT to really
        // speed up the verification procedure. 
        void initialize(const std::vector<std::vector<int>>& sample, const HybridSolver &solver,
                        int solver_id, ModelVector* estimated_models)
        {
            std::chrono::time_point<std::chrono::system_clock> end,
                start = std::chrono::system_clock::now();
            const int kNumEstimatedModels = solver.MinimalSolver(
                sample, solver_id, estimated_models);
            end = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsedSeconds = end - start;
            t_M = elapsedSeconds.count() * 1000.0; // The time of estimating a model

            if (kNumEstimatedModels > 0)
            {
                start = std::chrono::system_clock::now();
                for (int i = 0; i < solver.num_data(); ++i)
                    double error = solver.EvaluateModelOnPoint((*estimated_models)[0], solver_id, i);
                end = std::chrono::system_clock::now();
                t_M = (elapsedSeconds.count() * 1000.0 / solver.num_data()) / t_M; // The time of estimating a model
            }

            m_S = 1; // The maximum number of solutions

            /*printf("Setting up SPRT test.\n");
            printf("\tThe estimation of one models takes %f ms.\n", t_M);
            printf("\tAt most %.0f models are returned.\n", m_S);*/
        }

        SPRTPreemptiveVerification(const HybridSolver &solver, int solver_id,
            const std::vector<std::vector<int>>& sample,
            ModelVector* res, const double &minimum_inlier_ratio_ = 0.01)
        {
            std::vector<std::vector<int>> min_sample_sizes;
            solver.min_sample_sizes(&min_sample_sizes);
            if (solver.num_data() < min_sample_sizes[solver_id][0])
            {
                fprintf(stderr, "There are not enough points to initialize the SPRT test (%d < %d).\n",
                    solver.num_data(), min_sample_sizes[solver_id][0]);
                return;
            }

            initialize(sample, solver, solver_id, res);

            additional_model_probability = 1.0;
            const size_t point_number = solver.num_data();

            // Generate array of points
            points_random_pool = new size_t[point_number];
            for (size_t i = 0; i < point_number; i++) {
                points_random_pool[i] = i;
            }
            unsigned int temp;
            int max = point_number;
            // Shuffle random pool of points.
            for (unsigned int i = 0; i < point_number; i++) {
                random_pool_idx = rand() % max;
                temp = points_random_pool[random_pool_idx];
                max--;
                points_random_pool[random_pool_idx] = points_random_pool[max];
                points_random_pool[max] = temp;
            }
            random_pool_idx = 0;

            sprt_histories = std::vector<SPRTHistory>();
            sprt_histories.emplace_back(SPRTHistory());

            sprt_histories.back().delta = minimum_inlier_ratio_ / 10.0;
            sprt_histories.back().epsilon = minimum_inlier_ratio_;

            current_sprt_idx = 0;

            sprt_histories.back().A = estimateThresholdA(sprt_histories.back().epsilon, sprt_histories.back().delta);
            sprt_histories.back().k = 0;

            number_rejected_models = 0;
            sum_fraction_data_points = 0;

            max_hypothesis_test_before_sprt = 20;
        }

        /*
            *                      p(x(r)|Hb)                  p(x(j)|Hb)
            * lambda(j) = Product (----------) = lambda(j-1) * ----------
            *                      p(x(r)|Hg)                  p(x(j)|Hg)
            * Set j = 1
            * 1.  Check whether j-th data point is consistent with the
            * model
            * 2.  Compute the likelihood ratio λj eq. (1)
            * 3.  If λj >  A, decide the model is ’bad’ (model ”re-jected”),
            * else increment j or continue testing
            * 4.  If j = N the number of correspondences decide model ”accepted
            *
            * Verifies model and returns model score.
            */
        // bool verifyModel(const gcransac::Model &model_,
        //     const _ModelEstimator &estimator_, // The model estimator
        //     const double &threshold_,
        //     const size_t &iteration_number_,
        //     const Score &best_score_,
        //     const cv::Mat &points_,
        //     const size_t *minimal_sample_,
        //     const size_t sample_number_,
        //     std::vector<size_t> &inliers_,
        //     Score &score_,
        //     const std::vector<const std::vector<size_t>*> *index_sets_ = nullptr)
        double verifyModel(
            const HybridSolver& solver, const Model& model,
            double squared_threshold,
            int solver_id, const std::vector<int> num_data,
            double curr_best_score, double solver_weight
        )
        {
            const size_t &point_number = solver.num_data();
            const double &epsilon = sprt_histories[current_sprt_idx].epsilon;
            const double &delta = sprt_histories[current_sprt_idx].delta;
            const double &A = sprt_histories[current_sprt_idx].A;

            double lambda_new, lambda = 1;
            size_t tested_point = 0, tested_inliers = 0;
            int curr_inlier_number = 0;

            bool valid_model = true;
            double score = 0;
            for (tested_point = 0; tested_point < point_number; tested_point++)
            {
                int point_idx = points_random_pool[random_pool_idx];
                double squared_residual = solver.EvaluateModelOnPoint(model, solver_id, point_idx);
                score += std::min(squared_residual, squared_threshold) * solver_weight;

                // Inliers 
                if (squared_residual < squared_threshold) {
                    lambda_new = lambda * (delta / epsilon);

                    // Increase the inlier number
                    ++curr_inlier_number;
                }
                else {
                    lambda_new = lambda * ((1 - delta) / (1 - epsilon));
                }

                // Increase the pool pointer and reset if needed
                if (++random_pool_idx >= point_number)
                    random_pool_idx = 0;

                if ((lambda_new > A * additional_model_probability) || (score > curr_best_score)) {
                    valid_model = false;
                    ++tested_point;
                    break;
                }

                lambda = lambda_new;
            }

            if (valid_model)
            {
                /*
                    * Model accepted and the largest support so far:
                    * design (i+1)-th test (εi + 1= εˆ, δi+1 = δˆ, i = i + 1).
                    * Store the current model parameters θ
                */
                if (score < curr_best_score) {
                    SPRTHistory new_sprt_history;

                    new_sprt_history.epsilon = (double)curr_inlier_number / point_number;

                    new_sprt_history.delta = delta;
                    new_sprt_history.A = estimateThresholdA(new_sprt_history.epsilon, delta);
                    
                    new_sprt_history.k++;
                    ++current_sprt_idx;
                    sprt_histories.emplace_back(new_sprt_history);
                }
            }
            else
            {
                /*
                    * Since almost all tested models are ‘bad’, the probability
                    * δ can be estimated as the average fraction of consistent data points
                    * in rejected models.
                */
                double delta_estimated = (double)curr_inlier_number / point_number;

                if (delta_estimated > 0 && fabs(delta - delta_estimated) / delta > 0.05) 
                {
                    SPRTHistory new_sprt_history;

                    new_sprt_history.epsilon = epsilon;
                    new_sprt_history.delta = delta_estimated;
                    new_sprt_history.A = estimateThresholdA(epsilon, delta_estimated);
                    new_sprt_history.k++;
                    current_sprt_idx++;
                    sprt_histories.emplace_back(new_sprt_history);
                }

                score = std::numeric_limits<double>::max();
            }

            return score;
        }

        /*
        * A(0) = K1/K2 + 1
        * A(n+1) = K1/K2 + 1 + log (A(n))
        * K1 = t_M / P_g
        * K2 = m_S/(P_g*C)
        * t_M is time needed to instantiate a model hypotheses given a sample
        * P_g = epsilon ^ m, m is the number of data point in the Ransac sample.
        * m_S is the number of models that are verified per sample.
        *                   p (0|Hb)                  p (1|Hb)
        * C = p(0|Hb) log (---------) + p(1|Hb) log (---------)
        *                   p (0|Hg)                  p (1|Hg)
        */
        double estimateThresholdA(
            const double &epsilon, 
            const double &delta)
        {
            const double C = (1 - delta) * log((1 - delta) / (1 - epsilon)) + delta * (log(delta / epsilon));
            const double K = (t_M * C) / m_S + 1;
            double An_1 = K;
            double An;
            for (unsigned int i = 0; i < 10; ++i) {
                An = K + log(An_1);

                if (fabs(An - An_1) < 1.5e-8) {
                    break;
                }
                An_1 = An;
            }

            return An;
        }
    };

} // namespace uncalibrated_vp 

#endif
