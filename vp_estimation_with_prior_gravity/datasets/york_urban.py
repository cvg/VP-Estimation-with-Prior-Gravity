""" YorkUrbanDB dataset for VP and rotation estimation. """

import os
import cv2
import scipy.io
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def simple_collate_fn(sample):
    return sample


class YorkUrban(Dataset):
    def __init__(self, root_dir, split):
        assert split in ["val", "test"], "Only split accepted are val and test."
        self.root_dir = root_dir
        self.img_names = [name for name in sorted(os.listdir(self.root_dir))
                          if os.path.isdir(os.path.join(self.root_dir, name))]
        assert len(self.img_names) == 102

        # Separate validation and test
        split_file = os.path.join(self.root_dir,
                                  'ECCV_TrainingAndTestImageNumbers.mat')
        split_mat = scipy.io.loadmat(split_file)
        if split == 'val':
            valid_set = split_mat['trainingSetIndex'][:, 0] - 1
        else:
            valid_set = split_mat['testSetIndex'][:, 0] - 1
        self.img_names = np.array(self.img_names)[valid_set]
        assert len(self.img_names) == 51

        # Load the intrinsics
        K_file = os.path.join(self.root_dir, 'cameraParameters.mat')
        K_mat = scipy.io.loadmat(K_file)
        f = K_mat['focal'][0, 0] / K_mat['pixelSize'][0, 0]
        p_point = K_mat['pp'][0] - 1  # -1 to convert to 0-based conv
        self.K = np.array([[f, 0, p_point[0]],
                           [0, f, p_point[1]],
                           [0, 0, 1]])

    def get_dataloader(self):
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=True,
            num_workers=0, collate_fn=simple_collate_fn)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx],
                                f'{self.img_names[idx]}.jpg')
        name = str(Path(img_path).stem)
        img = cv2.imread(img_path, 0)

        # Load the GT lines and VP association
        lines_file = os.path.join(self.root_dir, self.img_names[idx],
                                  f'{self.img_names[idx]}LinesAndVP.mat')
        lines_mat = scipy.io.loadmat(lines_file)
        lines = lines_mat['lines'].reshape(-1, 2, 2)[:, :, [1, 0]] - 1
        vp_association = lines_mat['vp_association'][:, 0] - 1

        # Load the VPs (non orthogonal ones)
        vp_file = os.path.join(
            self.root_dir, self.img_names[idx],
            f'{self.img_names[idx]}GroundTruthVP_CamParams.mat')
        vps = scipy.io.loadmat(vp_file)['vp'].T

        # Load the orthogonal VPs
        vp_file = os.path.join(
            self.root_dir, self.img_names[idx],
            f'{self.img_names[idx]}GroundTruthVP_Orthogonal_CamParams.mat')
        orth_vps = scipy.io.loadmat(vp_file)['vp_orthogonal'].T

        # Compute the world to cam rotation matrix (second vp is vertical)
        R_world_to_cam = orth_vps[[2, 0, 1]].T

        return {'image': img, 'image_path': img_path, 'name': name,
                'gt_lines': lines, 'vps': vps, 'orth_vps': orth_vps,
                'vp_association': vp_association, 'K': self.K,
                'R_world_to_cam': R_world_to_cam}

    def __len__(self):
        return len(self.img_names)
