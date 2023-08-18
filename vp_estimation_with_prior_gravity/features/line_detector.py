import os
import numpy as np
import torch
from copy import deepcopy
from omegaconf import OmegaConf

from pytlsd import lsd
from deeplsd.models.deeplsd_inference import DeepLSD


# DeepLSD default config
default_conf = {
    'min_length': 10,
    'deeplsd': {
        'detect_lines': True,
        'line_detection_params': {
            'merge': False,
            'filtering': True,
            'grad_thresh': 3,
            'grad_nfa': True,
        },
    },
}


class LineDetector():
    """ A generic line detector that can detect different kinds of lines. """
    def __init__(self, line_detector='lsd', conf=default_conf):
        self.detector = line_detector
        self.conf = OmegaConf.merge(OmegaConf.create(deepcopy(default_conf)),
                                    OmegaConf.create(conf))
        if line_detector == 'deeplsd':
            # Load the DeepLSD model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/deeplsd_md.tar')
            if not os.path.isfile(ckpt):
                # Download the DeepLSD checkpoint
                self.download_model(ckpt)
            ckpt = torch.load(str(ckpt), map_location='cpu')
            self.deeplsd = DeepLSD(self.conf.deeplsd)
            self.deeplsd.load_state_dict(ckpt['model'])
            self.deeplsd = self.deeplsd.to(self.device).eval()

    def download_model(self, path):
        import subprocess
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        link = "https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys/download"
        cmd = ["wget", link, "-O", path]
        print("Downloading DeepLSD model...")
        subprocess.run(cmd, check=True)

    def detect_lines(self, image):
        """ Detect line segments and output them as a [N, 2, 2] np.array
            in row-col coordinates convention. """
        if self.detector == 'lsd':
            lines = lsd(
                image)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        elif self.detector == 'deeplsd':
            torch_img = torch.tensor(image[None, None], dtype=torch.float,
                                     device=self.device) / 255
            with torch.no_grad():
                lines = self.deeplsd({'image': torch_img})['lines'][0][:, :, [1, 0]]
        else:
            raise ValueError("Unknown line detector: " + self.detector)

        # Remove short lines
        line_lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        mask = line_lengths > self.conf.min_length
        lines = lines[mask]

        return lines
