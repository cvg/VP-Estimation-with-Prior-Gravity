import os
import numpy as np
import torch
from copy import deepcopy
from omegaconf import OmegaConf

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline


default_conf = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {
            'force_num_keypoints': False,
            'max_num_keypoints': 2000,
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
        },
        'max_n_lines': 500,
    },
    'matcher': {
        'name': 'gluestick',
        'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
        'trainable': False,
    },
    'ground_truth': {
        'from_pose_depth': False,
    }
}


class PointLineMatcher():
    """ Point-line matcher using GlueStick (https://github.com/cvg/GlueStick) """
    def __init__(self, conf=default_conf):
        self.conf = OmegaConf.merge(OmegaConf.create(deepcopy(default_conf)),
                                    OmegaConf.create(conf))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = self.conf.matcher.weights
        if not os.path.isfile(ckpt):
            # Download the GlueStick checkpoint
            self.download_model(ckpt)
        self.gluestick = TwoViewPipeline(self.conf).to(self.device).eval()

    def download_model(self, path):
        import subprocess
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar"
        cmd = ["wget", link, "-O", path]
        print("Downloading GlueStick model...")
        subprocess.run(cmd, check=True)

    def match_points_and_lines(self, img0, img1, line_features=None):
        # GlueStick forward pass
        torch_img0 = numpy_image_to_torch(img0).to(self.device)[None]
        torch_img1 = numpy_image_to_torch(img1).to(self.device)[None]
        inputs = {'image0': torch_img0, 'image1': torch_img1}
        if line_features is not None:
            inputs['lines0'] = torch.tensor(
                line_features['lines0'], dtype=torch.float,
                device=self.device)[None]
            inputs['lines1'] = torch.tensor(
                line_features['lines1'], dtype=torch.float,
                device=self.device)[None]
            inputs['line_scores0'] = torch.tensor(
                line_features['line_scores0'], dtype=torch.float,
                device=self.device)[None]
            inputs['line_scores1'] = torch.tensor(
                line_features['line_scores1'], dtype=torch.float,
                device=self.device)[None]
            inputs['valid_lines0'] = torch.ones(
                1, len(line_features['line_scores0']), dtype=torch.bool,
                device=self.device)
            inputs['valid_lines1'] = torch.ones(
                1, len(line_features['line_scores1']), dtype=torch.bool,
                device=self.device)
        with torch.no_grad():
            pred = self.gluestick(inputs)
            pred = batch_to_np(pred)

        # Get the point matches
        matches = pred['matches0']
        valid = matches > -1
        mkpts0 = pred['keypoints0'][valid]
        mkpts1 = pred['keypoints1'][matches[valid]]

        # Get the line matches
        lines0 = pred['lines0'][:, :, [1, 0]]
        lines1 = pred['lines1'][:, :, [1, 0]]
        line_matches = pred['line_matches0']
        valid = line_matches > -1
        mlines0 = lines0[valid]
        mlines1 = lines1[line_matches[valid]]

        return (np.concatenate([mkpts0, mkpts1], axis=1),
                lines0, lines1, mlines0, mlines1)
