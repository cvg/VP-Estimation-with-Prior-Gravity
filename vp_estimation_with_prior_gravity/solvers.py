import sys
sys.path.append("build/vp_estimation_with_prior_gravity")
import vp_estimation_with_prior_gravity_estimators as _estimators
import math
import numpy as np


def run_calibrated_3lines(lines, K, th_pixels=1.0, use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 3)
        func = _estimators.run_calibrated_3lines_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_calibrated_3lines
    res = func(lines.reshape(-1, 4).transpose(1,0), K, options)
    vps = res[0].T
    return vps


def run_calibrated_vertical_1line(lines, f, v, th_pixels=1.0, ls_refinement=0, nms=0, use_exhaustive=True):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.num_lo_steps_ = 20  # 100
    #options.num_lsq_iterations_ = 100
    options.lo_starting_iterations_ = 1
    if use_exhaustive:
        options.min_num_iterations_ = lines.shape[0]
        options.max_num_iterations_ = lines.shape[0]
        func = _estimators.run_calibrated_vertical_1line_exs
    else:
        options.min_num_iterations_ = 1000
        options.max_num_iterations_ = 1000
        func = _estimators.run_calibrated_vertical_1line
    res = func(lines.reshape(-1, 4).transpose(1,0), f, v, options, ls_refinement, nms)
    vps = res[0].T
    return vps


def run_uncalibrated_4lines_220(
    lines, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.non_min_sample_multiplier_ = 25
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 4)
        func = _estimators.run_uncalibrated_4lines_220_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_4lines_220
    res = func(lines.reshape(-1, 4).transpose(1,0), options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_uncalibrated_4lines_211(
    lines, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.non_min_sample_multiplier_ = 25
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 4)
        func = _estimators.run_uncalibrated_4lines_211_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_4lines_211
    res = func(lines.reshape(-1, 4).transpose(1,0), options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_uncalibrated_vertical_2lines_2(
    lines, v, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.num_lo_steps_ = 20  # 100
    options.non_min_sample_multiplier_ = 25  # 20 with GT gravity, 25 without
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 2)
        func = _estimators.run_uncalibrated_vertical_2lines_2_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_vertical_2lines_2
    res = func(lines.reshape(-1, 4).transpose(1,0), v, options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_uncalibrated_vertical_2lines_11(
    lines, v, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.num_lo_steps_ = 20  # 100
    options.non_min_sample_multiplier_ = 25  # 20 with GT gravity, 25 without
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 2)
        func = _estimators.run_uncalibrated_vertical_2lines_11_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_vertical_2lines_11
    res = func(lines.reshape(-1, 4).transpose(1,0), v, options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_uncalibrated_vertical_2lines_11_v2(
    lines, v, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.num_lo_steps_ = 20  # 100
    options.non_min_sample_multiplier_ = 25  # 20 with GT gravity, 25 without
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 2)
        func = _estimators.run_uncalibrated_vertical_2lines_11_v2_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_vertical_2lines_11_v2
    res = func(lines.reshape(-1, 4).transpose(1,0), v, options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_uncalibrated_vertical_2lines_011(
    lines, v, th_pixels=1.0, ls_refinement=0, nms=0, magsac_scoring=False,
    use_exhaustive=False):
    options = _estimators.LORansacOptions()
    options.squared_inlier_threshold_ = th_pixels
    options.num_lo_steps_ = 20  # 100
    options.non_min_sample_multiplier_ = 25  # 20 with GT gravity, 25 without
    if use_exhaustive:
        options.min_num_iterations_ = math.comb(lines.shape[0], 2)
        func = _estimators.run_uncalibrated_vertical_2lines_011_exs
    else:
        options.min_num_iterations_ = 1000
        func = _estimators.run_uncalibrated_vertical_2lines_011
    res = func(lines.reshape(-1, 4).transpose(1,0), v, options, ls_refinement,
               nms, magsac_scoring)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps


def run_hybrid_uncalibrated(lines, v, th_pixels=1.0, ls_refinement=0, nms=0,
                            magsac_scoring=False, sprt=False,
                            solver_flags=[True, True, True, True, True]):
    """
    Solvers
    0 - 2lines 2
    1 - 2lines 11 v2
    2 - 2lines 011
    3 - 4lines 211
    4 - 4lines 220
    """
    options = _estimators.ExtendedHybridLORansacOptions()
    options.squared_inlier_thresholds_ = [th_pixels]
    options.data_type_weights_ = [1.0]
    options.num_lo_steps_ = 20  # 100
    options.non_min_sample_multiplier_ = 25  # 20 with GT gravity, 25 without
    options.min_num_iterations_ = 1000 // np.array(solver_flags, dtype=int).sum()

    func = _estimators.run_hybrid_uncalibrated
    res = func(lines.reshape(-1, 4).transpose(1,0), v, options,
               ls_refinement, nms, magsac_scoring, sprt, solver_flags)
    focal_length = res[0][0]
    vps = res[0][1].T
    return focal_length, vps
