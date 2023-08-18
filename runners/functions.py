import numpy as np
import h5py
from scipy.optimize import linear_sum_assignment

from vp_estimation_with_prior_gravity.evaluation import dist_lines_vp


def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    try:
        #with self.lock:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load


def append_h5(dict_to_save, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'a') as f:
        #with self.lock:
        for key in dict_to_save:
            if key in f.keys():
                del f[key]
            f.create_dataset(key, data=dict_to_save[key])


def read_h5(key, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'a') as f:
        #with self.lock:
        if key in f.keys():
            return np.array(f.get(key))
        else:
            return None


def project_vps_to_image(vp, K):
    """ Convert 3D directions to VPs in homogenous format in the image plane. """
    proj_vp = (K @ vp.T).T
    norm = np.linalg.norm(proj_vp, axis=1, keepdims=True)
    proj_vp *= np.where(norm > 1e-5, 1. / norm, np.ones_like(proj_vp))
    return proj_vp


def vp_matching(vp0, label0, vp1, label1):
    """ Match the vanishing points between two images,
        given a corresponding set of lines. The lines corresponding
        to label0 and label1 should be matching. """
    assert len(label0) == len(label1), "1:1 correspondence between lines is not respected."

    # Build a matrix of common lines
    n_vp0, n_vp1 = len(vp0), len(vp1)
    if n_vp0 == 0 or n_vp1 == 0:
        m_vp0, m_vp1 = [], []
        m_label0, m_label1 = -np.ones_like(label0), -np.ones_like(label1)
        return m_vp0, m_label0, m_vp1, m_label1

    common_lines = np.zeros((n_vp0, n_vp1))
    for l0, l1 in zip(label0, label1):
        if l0 == -1 or l1 == -1:
            continue
        common_lines[l0, l1] += 1

    # Compute the optimal assignment
    row_assignment, col_assignment = linear_sum_assignment(common_lines,
                                                           maximize=True)

    # Get the matched VPs and new labeling
    # -1 means that a line is not shared by the same VP (or has no VP)
    m_vp0 = vp0[row_assignment]
    m_vp1 = vp1[col_assignment]
    m_label0, m_label1 = -np.ones_like(label0), -np.ones_like(label1)
    for i in range(len(row_assignment)):
        m_label0[label0 == row_assignment[i]] = i
        m_label1[label1 == col_assignment[i]] = i

    return m_vp0, m_label0, m_vp1, m_label1


def match_vps(vp1, vp2, m_lines1, m_lines2, K1, K2, tol_px=5):
    """ Match two sets of vps (given in world coordinates),
        based on lines matches. """
    # Project VPs to the image
    img_vp1 = project_vps_to_image(vp1, K1)
    img_vp2 = project_vps_to_image(vp2, K2)

    # Assign each match line to the closest VP (or none if it is too far away)
    line_vp_dist1 = dist_lines_vp(m_lines1, img_vp1)
    closest1 = np.argmin(line_vp_dist1, axis=1)
    closest1[np.amin(line_vp_dist1, axis=1) > tol_px] = -1
    line_vp_dist2 = dist_lines_vp(m_lines2, img_vp2)
    closest2 = np.argmin(line_vp_dist2, axis=1)
    closest2[np.amin(line_vp_dist2, axis=1) > tol_px] = -1

    # Match the VPs
    m_vp1, label1, m_vp2, label2 = vp_matching(vp1, closest1, vp2, closest2)
    return m_vp1, m_vp2, label1, label2
