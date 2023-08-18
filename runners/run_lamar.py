#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from vp_estimation_with_prior_gravity.datasets.lamar import Lamar
from vp_estimation_with_prior_gravity.features.line_detector import LineDetector
from vp_estimation_with_prior_gravity.features.matcher import PointLineMatcher
from vp_estimation_with_prior_gravity.evaluation import evaluate_R
from vp_estimation_with_prior_gravity.solvers import (
    run_calibrated_3lines, run_calibrated_vertical_1line,
    run_hybrid_uncalibrated)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import append_h5, read_h5, match_vps
from evaluator import Evaluator


# Parameters
output_dir = "experiments/lamar/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
use_exhaustive = False  # Use the exhaustive sampler
num_runs = 5

# Which solvers to us for our hybrid solver:
# 0 - 2lines 200
# 1 - 2lines 110
# 2 - 2lines 011
# 3 - 4lines 211
# 4 - 4lines 220
SOLVER_FLAGS = [True, True, True, True, True]

# RANSAC thresholds (obtained from the validation set)
th_pixels = {
    "calibrated_210": 1.,
    "calibrated_100g": 2.,
    "uncalibrated_220": 3.,
    "uncalibrated_211": 3.,
    "uncalibrated_200g": 1.,
    "uncalibrated_110g": 3.,
    "uncalibrated_011g": 3.,
    "hybrid": 3.5,
}


def vps_to_rel_rot(vps):
    # Given two sets of 3 orthogonal VPs, make sure the VPs have the
    # same direction and compute the relative rotation between them
    # The rotations are from world to cam
    R1 = np.stack(vps[0], axis=-1)
    if np.linalg.det(R1) < 0:
        R1[:, 2] *= -1
    R2 = np.stack(vps[1], axis=-1)
    if np.linalg.det(R2) < 0:
        R2[:, 2] *= -1
    R2 *= np.sign((R1 * R2).sum(axis=0)).reshape(1, 3)
    return R2 @ R1.T


def detect_and_load_data(data, line_detector, matcher, match_db_file):
    # Detect and match points and lines
    img1 = data["img1"]
    img2 = data["img2"]
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Database labels
    label1 = "-".join(data["id1"].split("/")[-3:])
    label2 = "-".join(data["id2"].split("/")[-3:])
    label = f"{line_detector}-{label1}-{label2}"

    lines1 = read_h5(f"{label}-lines-1", match_db_file)
    lines2 = read_h5(f"{label}-lines-2", match_db_file)
    point_matches = read_h5(f"point_matches-{label1}-{label2}", match_db_file)
    m_lines1 = read_h5(f"{label}-mlines-1", match_db_file)
    m_lines2 = read_h5(f"{label}-mlines-2", match_db_file)
    if (lines1 is None or lines2 is None or point_matches is None
        or m_lines1 is None or m_lines2 is None):
        if line_detector.detector == 'deeplsd':
            # Detect DeepLSD lines
            lines1 = line_detector.detect_lines(gray_img1)[:, :, [1, 0]]
            lines2 = line_detector.detect_lines(gray_img2)[:, :, [1, 0]]
            line_scores1 = np.sqrt(np.linalg.norm(
                lines1[:, 0] - lines1[:, 1], axis=1))
            line_scores2 = np.sqrt(np.linalg.norm(
                lines2[:, 0] - lines2[:, 1], axis=1))
            line_features = {
                'lines0': lines1, 'lines1': lines2,
                'line_scores0': line_scores1, 'line_scores1': line_scores2
            }
        else:
            # GlueStick already detects LSD lines by default
            line_features = None
        
        # Detect points and match all features together
        point_matches, lines1, lines2, m_lines1, m_lines2 = matcher.match_points_and_lines(
            gray_img1, gray_img2, line_features)

        # Saving to the database
        append_h5({f"point_matches-{label1}-{label2}": point_matches}, match_db_file)
        append_h5({f"{label}-mlines-1": m_lines1,
                   f"{label}-mlines-2": m_lines2,
                   f"{label}-lines-1": lines1,
                   f"{label}-lines-2": lines2},
                   match_db_file)

    return point_matches, lines1, lines2, m_lines1, m_lines2


def run_lamar(args):
    # Initialize some parameters
    use_gt_gravity = args.use_gt_gravity
    ls_refinement = 3 if use_gt_gravity else 2  # 3 uses the gravity in the LS refinement, 2 does not
    # 0 disables the VP refinement, and 1 is a LS for non orthogonal VPs
    if args.nms == 'none':
        nms = 0
    elif args.nms == 'non_orth':
        nms = 1
    elif args.nms == 'ceres':
        nms = 3
    else:
        raise ValueError("Unknown non minimal solver: " + args.nms)
    sprt = not args.disable_sprt
    magsac_scoring = args.magsac_scoring
    solver = args.solver
    match_db_file = f"{output_dir}matches_{args.line_detector}.h5"

    # Initialize dataset and evaluator
    dataset = Lamar(root_dir=args.data_root)
    dataloader = dataset.get_dataloader()
    evaluator = Evaluator(vp_metrics=False, use_rot=True, uncalibrated=True)

    line_detector = LineDetector(line_detector=args.line_detector)
    matcher = PointLineMatcher()
    evaluator.add_method(solver)
    for data in tqdm(dataloader):
        # GT data
        img1 = cv2.cvtColor(data['img1'], cv2.COLOR_RGB2GRAY)
        h, w = img1.shape
        K1 = data['K1']
        K2 = data['K2']
        R_1_2 = data['R_1_2']
        R1_gt = data['R_world_to_cam1']
        R2_gt = data['R_world_to_cam2']
        if use_gt_gravity:
            vertical1 = -R1_gt[:, 2]
            vertical2 = -R2_gt[:, 2]
        else:
            vertical1 = data['gravity1']
            vertical2 = data['gravity2']

        # Obtain the principal point
        if solver.startswith("calibrated"):
            principle_point = np.array([K[0, 2], K[1, 2]])
        else:
            principle_point = np.array([w / 2.0, h / 2.0])

        # Detect and match lines
        _, lines1, lines2, m_lines1, m_lines2 = detect_and_load_data(
            data, line_detector, matcher, match_db_file)

        # Estimate the relative rotation for a given number of runs
        for _ in range(num_runs):
            pred_vps, pred_f = [], []
            for lines, K, vertical in zip(
                [lines1, lines2], [K1, K2], [vertical1, vertical2]):
                if solver == "calibrated_3lines":
                    vp = run_calibrated_3lines(lines[:, :, [1, 0]], K,
                                               th_pixels=th_pixels[solver])
                elif solver == "calibrated_100g":
                    f = (K[0, 0] + K[1, 1]) / 2
                    vp = run_calibrated_vertical_1line(
                        lines[:, :, [1, 0]] - principle_point, f,
                        vertical, th_pixels=th_pixels[solver],
                        ls_refinement=ls_refinement, nms=nms, use_exhaustive=use_exhaustive)
                elif solver == "uncalibrated_220":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=[False, False, False, False, True])
                elif solver == "uncalibrated_211":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=[False, False, False, True, False])
                elif solver == "uncalibrated_200g":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=[True, False, False, False, False])
                elif solver == "uncalibrated_110g":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=[False, True, False, False, False])
                elif solver == "uncalibrated_011g":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=[False, False, True, False, False])
                elif solver == "hybrid":
                    f, vp = run_hybrid_uncalibrated(
                        lines[:, :, [1, 0]] - principle_point[None, None, :],
                        vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                        nms=nms, sprt=sprt, magsac_scoring=magsac_scoring,
                        solver_flags=SOLVER_FLAGS)
                else:
                    raise ValueError("Unknown method: " + solver)
                pred_vps.append(vp)
                if solver.startswith("uncalibrated") or solver == "hybrid":
                    pred_f.append(f)

            # Match the VPs
            m_vp1, m_vp2, _, _ = match_vps(pred_vps[0], pred_vps[1],
                                           m_lines1, m_lines2, K1, K2)
            pred_vps = [m_vp1, m_vp2]

            # Compute the relative rotation error
            pred_R = vps_to_rel_rot(pred_vps)
            rot_err = evaluate_R(R_1_2, pred_R)
            evaluator.add_rot_error_multirun(solver, rot_err)

        if solver.startswith("uncalibrated") or solver == "hybrid":
            # Focal length estimation evaluation
            gt_f1 = (K1[0, 0] + K1[1, 1]) / 2
            evaluator.add_f_error_multirun(solver, np.abs(pred_f[0] - gt_f1) / gt_f1)
            gt_f2 = (K2[0, 0] + K2[1, 1]) / 2
            evaluator.add_f_error_multirun(solver, np.abs(pred_f[1] - gt_f2) / gt_f2)
        evaluator.end_multirun(solver)

    # Store the results on disk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fname = os.path.join(output_dir, "{0}.npz".format(solver))
    evaluator.save_report(fname, solver)

    # Display the results
    evaluator.report()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Rotation estimation on Lamar.')
    parser.add_argument(
        'solver', type=str, default='hybrid',
        help="Type of Minimal Solver: 'calibrated_210' for calibrated images with 3 lines, 'calibrated_100g' for calibrated images with 1 line and gravity, 'uncalibrated_220' for uncalibrated images with 4 lines (220), ''uncalibrated_211' for uncalibrated images with 4 lines (211), 'uncalibrated_200g' for uncalibrated images with 2 lines and gravity (200), 'uncalibrated_011g' for uncalibrated images with 2 lines and gravity (011), 'uncalibrated_110g' for uncalibrated images with 2 lines and gravity 110), or 'hybrid' for our hybrid solver combining all uncalibrated ones.")
    parser.add_argument(
        'data_root', type=str, help="Path to the dataset.")
    parser.add_argument(
        '--nms', type=str, default='non_orth',
        help="Type of Non Minimal Solver: 'none', 'non_orth' for our proposed one, or 'ceres' for our proposed one followed by Ceres optimization (slower).")
    parser.add_argument(
        '--line_detector', type=str, default='lsd',
        help="Type of line detector: 'lsd' or 'deeplsd'.")
    parser.add_argument(
        '--use_gt_gravity', action='store_true',
        help='Use the ground truth gravity instead of the prior (default: False).')
    parser.add_argument(
        '--disable_sprt', action='store_true',
        help='Disable preemptive stopping (will be slower).')
    parser.add_argument(
        '--magsac_scoring', action='store_true',
        help='Use MAGSAC scoring.')
    args = parser.parse_args()

    # Run the evaluation on Lamar
    run_lamar(args)
