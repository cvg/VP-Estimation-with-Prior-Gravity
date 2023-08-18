#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import h5py

from vp_estimation_with_prior_gravity.datasets.york_urban import YorkUrban
from vp_estimation_with_prior_gravity.features.line_detector import LineDetector
from vp_estimation_with_prior_gravity.evaluation import (project_vp_to_image,
                                                         evaluate_R, vp2rot)
from vp_estimation_with_prior_gravity.solvers import (
    run_calibrated_3lines, run_calibrated_vertical_1line,
    run_hybrid_uncalibrated)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluator import Evaluator


# Parameters
output_dir = "experiments/york_urban/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
use_orthogonal_gt_vps = True  # Use the orthogonal GT VPs
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
    "hybrid": 3,
}


def run_york_urban(args):
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
    line_db_file = f"{output_dir}lines_{args.line_detector}.h5"

    # Initialize dataset and evaluator
    dataset = YorkUrban(root_dir=args.data_root, split='test')
    dataloader = dataset.get_dataloader()
    uncalibrated = solver.startswith("uncalibrated") or solver == "hybrid"
    evaluator = Evaluator(use_rot=(ls_refinement != 1), uncalibrated=uncalibrated)

    line_detector = LineDetector(line_detector=args.line_detector)
    for data in tqdm(dataloader):
        # GT data
        img = data['image']
        img_name = data['name']
        K = data['K']
        if use_orthogonal_gt_vps:
            gt_vp = data['orth_vps']
        else:
            gt_vp = data['vps']
        R_gt = data['R_world_to_cam']
        if use_gt_gravity:
            vertical = gt_vp[1]
        else:
            if (solver == "uncalibrated_200g"
                or solver == "uncalibrated_011g"
                or (solver == "hybrid" and (SOLVER_FLAGS == [True, False, False, False, False]))
                or (solver == "hybrid" and (SOLVER_FLAGS == [False, False, True, False, False]))):
                vertical = np.array([random.random() / 1e12, -1, random.random() / 1e12])
                vertical /= np.linalg.norm(vertical)
            else:
                vertical = np.array([0., -1, 0.])
        vertical = vertical * np.array([1., -1., 1.])

        # Obtain the principal point
        if solver.startswith("calibrated"):
            principle_point = np.array([K[0, 2], K[1, 2]])
        else:
            principle_point = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])

        # Regress line segments
        hf = h5py.File(line_db_file, 'a')
        if img_name in hf:
            lines = np.array(hf[img_name])
        else:
            lines = line_detector.detect_lines(img)
            hf.create_dataset(img_name, data=lines)
        hf.close()

        # Regress VPs with the given solver for num_runs runs
        for _ in range(num_runs):
            if solver == "calibrated_210":
                vp = run_calibrated_3lines(lines[:, :, [1, 0]], K, th_pixels=th_pixels[solver])
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
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=[False, False, False, False, True])
            elif solver == "uncalibrated_211":
                f, vp = run_hybrid_uncalibrated(
                    lines[:, :, [1, 0]] - principle_point[None, None, :],
                    vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=[False, False, False, True, False])
            elif solver == "uncalibrated_200g":
                f, vp = run_hybrid_uncalibrated(
                    lines[:, :, [1, 0]] - principle_point[None, None, :],
                    vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=[True, False, False, False, False])
            elif solver == "uncalibrated_110g":
                f, vp = run_hybrid_uncalibrated(
                    lines[:, :, [1, 0]] - principle_point[None, None, :],
                    vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=[False, True, False, False, False])
            elif solver == "uncalibrated_011g":
                f, vp = run_hybrid_uncalibrated(
                    lines[:, :, [1, 0]] - principle_point[None, None, :],
                    vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=[False, False, True, False, False])
            elif solver == "hybrid":
                f, vp = run_hybrid_uncalibrated(
                    lines[:, :, [1, 0]] - principle_point[None, None, :],
                    vertical, th_pixels=th_pixels[solver], ls_refinement=ls_refinement,
                    nms=nms, magsac_scoring=magsac_scoring, sprt=sprt,
                    solver_flags=SOLVER_FLAGS)
            else:
                raise ValueError("Unknown solver: " + solver)
            vp[:, 1] *= -1 # for york urban
            if ls_refinement != 1:
                pred_R = vp2rot(vp, R_gt)
            vp = project_vp_to_image(vp, K)
            evaluator.add_entry_multirun(solver, vp, gt_vp, K)
            if ls_refinement != 1:
                # Don't evaluate the rotation error if the VPs are not orthogonal
                if pred_R is None:
                    rot_err = 90  # Default rotation error
                else:
                    rot_err = evaluate_R(R_gt, pred_R)
                evaluator.add_rot_error_multirun(solver, rot_err)
            if uncalibrated:
                # Focal length estimation evaluation
                gt_f = (K[0, 0] + K[1, 1]) / 2
                evaluator.add_f_error_multirun(solver, np.abs(f - gt_f) / gt_f)
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
    parser = argparse.ArgumentParser(description='VP evaluation on York Urban.')
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

    # Run the evaluation on York Urban
    run_york_urban(args)
