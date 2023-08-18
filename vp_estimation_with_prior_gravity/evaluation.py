import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


### VP estimation metrics
def dist_lines_vp(lines, vp):
    """ Estimate the distance between a set of lines
        and VPs in homogeneous format.
    Args:
        lines: [N, 2, 2] array in ij convention.
        vp: [M, 3] array in homogeneous format.
    Returns:
        An [N, M] distance matrix of each line to each VP.
    """
    # Center of the lines
    centers = ((lines[:, 0] + lines[:, 1]) / 2)
    centers = np.concatenate([centers[:, [1, 0]],
                              np.ones_like(centers[:, :1])], axis=1)

    # Line passing through the VP and the center of the lines
    # l = cross(center, vp)
    # l is [N, M, 3]
    line_vp = np.cross(centers[:, None], vp[None])
    line_vp_norm = np.linalg.norm(line_vp[:, :, :2], axis=2)

    # Orthogonal distance of the lines to l
    endpts = np.concatenate([lines[:, 0][:, [1, 0]],
                             np.ones_like(lines[:, 0, :1])], axis=1)
    orth_dist = np.abs(np.sum(endpts[:, None] * line_vp,
                              axis=2))
    orth_dist[line_vp_norm < 1e-4] = 0
    line_vp_norm[line_vp_norm < 1e-4] = 1
    orth_dist /= line_vp_norm
    return orth_dist

def vp_consistency_check(gt_lines, line_clusters, vps, tol=3):
    """ Given a set of GT lines, their GT VP clustering and estimated VPs,
        assign each cluster of line to a unique VP and compute the ratio
        of lines consistent with the assigned VP.
        Return a list of consistency, for each tolerance threshold in tol. """
    if not isinstance(tol, list):
        tol = [tol]

    # Compute the distance from all lines to all VPs
    distances = dist_lines_vp(gt_lines, vps)

    # Compute the average score for each cluster of lines
    num_vps = len(vps)
    num_lines = len(gt_lines)
    cluster_labels = np.unique(line_clusters)
    num_clusters = len(cluster_labels)
    avg_scores = np.zeros((num_clusters, num_vps))
    for i in range(num_clusters):
        curr_cluster = line_clusters == cluster_labels[i]
        avg_scores[i] = np.mean(distances[curr_cluster], axis=0)

    # Find the optimal assignment of clusters and VPs
    cluster_assignment, vp_assignment = linear_sum_assignment(avg_scores)

    # Compute the number of consistent lines within each cluster
    consistency_check = []
    for t in tol:
        num_consistent = 0
        for cl, vp in zip(cluster_assignment, vp_assignment):
            num_consistent += np.sum(
                distances[line_clusters == cluster_labels[cl]][:, vp] < t)
        consistency_check.append(num_consistent / num_lines)

    return consistency_check

def project_vp_to_image(vp, K):
    """ Convert the VPs to homogenous format in the image plane. """
    eps = 1e-15
    proj_vp = vp / (vp[:, 2:] + eps)
    proj_vp[:, 1] *= -1
    proj_vp = (K @ proj_vp.T).T
    norm = np.linalg.norm(proj_vp, axis=1)
    proj_vp[norm > 1e-5] /= np.linalg.norm(proj_vp[norm > 1e-5], axis=1, keepdims=True)
    return proj_vp

def unproject_vp_to_world(vp, K):
    """ Convert the VPs from homogenous format in the image plane
        to world direction. """
    proj_vp = (np.linalg.inv(K) @ vp.T).T
    proj_vp[:, 1] *= -1
    proj_vp /= np.linalg.norm(proj_vp, axis=1, keepdims=True)
    return proj_vp

def get_labels_from_vp(lines, vp, threshold=1.0):
    dists = dist_lines_vp(lines, vp)
    min_value = np.min(dists, 1)
    mask = min_value < threshold
    indexes = np.argmin(dists, 1)
    labels = np.ones((lines.shape[0])) * -1
    labels[mask] = indexes[mask]
    return labels.astype(np.int64), dists

def get_vp_accuracy(gt_vp, pred_vp, K, max_err=10.):
    """ Compute the angular error between the predicted and GT VPs in 3D.
        The GT VPs are expected in 3D and unit normalized,
        but the predicted ones are in homogeneous format in the image. """
    # Unproject the predicted VP to world coordinates
    if pred_vp is None:
        return max_err
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum('nd,md->nm', gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)
    num_found = len(gt_idx)
    num_vp = len(gt_vp)

    # Get the accuracy in degrees
    accuracy = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    accuracy = np.arccos(accuracy) * 180 / np.pi

    # Clip to a maximum error of 10 degrees and penalize missing VPs
    accuracy = np.clip(accuracy, 0, max_err)
    accuracy = np.concatenate([accuracy,
                               max_err * np.ones(num_vp - num_found)])
    return np.mean(accuracy)

def get_vp_detection_ratio(gt_vp, pred_vp, K, thresholds):
    """ Compute the angular error between the predicted and GT VPs in 3D.
        The GT VPs are expected in 3D and unit normalized,
        but the predicted ones are in homogeneous format in the image.
        Count how many correct VPs are obtained for each error threshold. """
    # no prediction
    if pred_vp is None:
        return [0 for t in thresholds]

    # Unproject the predicted VP to world coordinates
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum('nd,md->nm', gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)

    # Get the accuracy in degrees
    num_gt_vp = len(gt_vp)
    accuracy = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    accuracy = np.arccos(accuracy) * 180 / np.pi
    scores = [np.sum(accuracy < t) / num_gt_vp for t in thresholds]
    return scores


def get_recall_AUC(gt_vp, pred_vp, K):
    """ Compute the angular error between the predicted and GT VPs in 3D,
        compute the recall for different error thresholds, and compute the AUC.
        The GT VPs are expected in 3D and unit normalized,
        but the predicted ones are in homogeneous format in the image. """
    # error thresholds
    step = 0.5
    error_thresholds = np.arange(0, 10.1, step=step)

    # no prediction
    if pred_vp is None:
        return [0 for t in error_thresholds], [0 for t in error_thresholds]

    # Unproject the predicted VP to world coordinates
    pred_vp_3d = pred_vp.copy()
    finite = np.abs(pred_vp_3d[:, 2]) > 1e-5
    pred_vp_3d[finite] /= pred_vp_3d[:, 2:][finite]
    pred_vp_3d = unproject_vp_to_world(pred_vp_3d, K)

    # Compute the pairwise cosine distances
    vp_dist = np.abs(np.einsum('nd,md->nm', gt_vp, pred_vp_3d))

    # Find the optimal assignment
    gt_idx, pred_idx = linear_sum_assignment(vp_dist, maximize=True)
    num_vp = len(gt_vp)

    # Get the accuracy in degrees
    accuracy = np.clip(vp_dist[gt_idx, pred_idx], 0, 1)
    accuracy = np.arccos(accuracy) * 180 / np.pi

    # Compute the recall at various error thresholds
    recalls = [np.sum(accuracy < t) / num_vp for t in error_thresholds]

    # Compute the AUC
    auc = np.sum(recalls) * step
    return recalls, auc


### Rotation estimation metrics

def vp2rot(vps, gt_R):
    """ Convert a set of 3 orthogonal VPs to a rotation matrix. """
    assert vps.shape == (3, 3), "vp2rot expects exactly 3 VPs"
    assert np.allclose(vps @ vps.T, np.eye(3), rtol=1e-4, atol=1e-4), "The vps should be orthogonal and unit normalized"
    # Find the best assignment between the vps and the GT ones
    vp_dist = (vps @ gt_R).T
    gt_idx, pred_idx = linear_sum_assignment(np.abs(vp_dist), maximize=True)
    rot = vps[pred_idx].T

    # Make the signs of the vector compatible
    rot *= np.sign(vp_dist[gt_idx, pred_idx][None])
    return rot


def get_E_from_F(F, K1, K2):
    return np.matmul(np.matmul(K2.T, F), K1)


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def eval_essential_matrix(p1n, p2n, E, R_gt):
    if len(p1n) != len(p2n):
        raise RuntimeError('Size mismatch in the keypoint lists')

    if p1n.shape[0] < 5:
        return 180

    if E.size > 0:
        _, R, t, _ = cv2.recoverPose(E, p1n, p2n)
        try:
            err_q = evaluate_R(R_gt, R)
        except:
            err_q = 180
    else:
        err_q = 180

    return err_q


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)
    return q


def qvec2rotmat(qvec):
    """ Convert from quaternions to rotation matrix. """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def evaluate_R(R_gt, R):
    eps = 1e-15
    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.rad2deg(np.arccos(1 - 2 * loss_q))

    if np.sum(np.isnan(err_q)):
        raise ValueError("NaN error.")

    return err_q


def pose_auc(errors, thresholds=[5, 10, 20]):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

