#ifndef UNCALIBRATED_VP_ESTIMATORS_EXHAUSTIVE_SAMPLER_H_
#define UNCALIBRATED_VP_ESTIMATORS_EXHAUSTIVE_SAMPLER_H_

#include <iostream>

namespace uncalibrated_vp {

inline unsigned nChoosek(unsigned n, unsigned k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for (int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;                    
    }
    return result;
}

// Exhaustive sampler that shares interface with UniformSampling in RansacLib
template <class Solver>
class ExhaustiveSampling {
public:
    ExhaustiveSampling(const unsigned int random_seed, const Solver& solver) {
        num_data_ = solver.num_data();
        sample_size_ = solver.min_sample_size();
        cur_indexes_.resize(sample_size_);
        reset_indexes();
    }

    void Sample(std::vector<int>* sample) {
        sample->resize(sample_size_);
        for (size_t i = 0; i < sample_size_; ++i) {
            (*sample)[i] = cur_indexes_[i];
        }
        update_indexes();
    }
    
    int NumCombinations() {
        return nChoosek(num_data_, sample_size_);
    }

private:
    int num_data_;
    int sample_size_;
    std::vector<int> cur_indexes_;

    void reset_indexes() {
        for (int i = 0; i < sample_size_; ++i) {
            cur_indexes_[i] = i;
        }
    }
    void update_indexes() {
        int now = sample_size_ - 1;
        while (true) {
            if (now == sample_size_ - 1) {
                if (cur_indexes_[now] < num_data_ - 1) {
                    cur_indexes_[now]++;
                    break;
                }
                else
                    now--;
            }
            else {
                if (cur_indexes_[now] < cur_indexes_[now + 1] - 1) {
                    cur_indexes_[now]++;
                    for (size_t i = now + 1; i < sample_size_; ++i)
                        cur_indexes_[i] = cur_indexes_[i - 1] + 1;
                    break;
                }
                else
                    now--;
            }
            // finish all the combinations
            if (now < 0) {
                reset_indexes();
                break;
            }
        }
    }
};

} // namespace uncalibrated_vp 

#endif

