""" ScanNet dataset for rotation estimation, as used in NeurVPS:
    https://github.com/zhou13/neurvps. """

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


K = np.array([[2.408333333333333 * 256, 0, 256],
              [0, 2.408333333333333 * 256, 256],
              [0, 0, 1]])


def simple_collate_fn(sample):
    return sample


class ScanNet(Dataset):
    def __init__(self, root_dir, split):
        assert split in ["val", "test"], "Only splits currently accepted are val and test."
        np.random.seed(0)
        # For each scene, extract 10 images randomly for test and one for val
        scenes = [f"scene0{s}_00" for s in range(707, 807)]
        self.img_files, self.vp_files, self.K = [], [], []
        for s in scenes:
            scene_path = os.path.join(root_dir, 'scans_test2', s)
            img_files = [os.path.join(scene_path, f)
                         for f in os.listdir(scene_path) if f.endswith('.png')]
            img_files.sort()
            vp_files = [os.path.join(scene_path, f)
                        for f in os.listdir(scene_path) if f.endswith('.npz')]
            vp_files.sort()
            
            if split == 'val':
                # Take the last frame
                self.img_files.append(img_files[-1])
                self.vp_files.append(vp_files[-1])
                self.K.append(K)
            else:
                # Randomly sample 10 frames
                selected_indices = np.random.choice(len(img_files) - 1, 10,
                                                    replace=False)
                self.img_files += np.array(img_files)[selected_indices].tolist()
                self.vp_files += np.array(vp_files)[selected_indices].tolist()
                self.K += [K] * 10

    def get_dataloader(self, shuffle=False):
        return DataLoader(
            self, batch_size=None, shuffle=shuffle, pin_memory=True,
            num_workers=4, collate_fn=simple_collate_fn)

    def __getitem__(self, item):
        # Read the image
        img = cv2.imread(self.img_files[item], 0)

        # Read the GT VPs
        vps = np.load(self.vp_files[item])
        vps = np.stack([vps['x'], vps['y'], vps['z']], axis=0)
        vps[:, 1] *= -1
        R_world_to_cam = vps.T

        outputs = {
            'image': img,
            'img_file': self.img_files[item],
            'vps': vps,
            'R_world_to_cam': R_world_to_cam,
            'K': self.K[item],
        }

        return outputs

    def __len__(self):
        return len(self.img_files)
