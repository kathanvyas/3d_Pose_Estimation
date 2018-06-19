# @Author: Kathan Vyas <Kathanvyas>
# @Date:   2018-06-17T16:33:33-04:00
# @Email:  kathan@usa.com
# @Last modified by:   Kathanvyas
# @Last modified time: 2018-06-19T01:18:26-04:00

import os
os.chdir('C:/Users/katha/Desktop/pose/tf-openpose/')
import numpy as np
import scipy
import scipy.io as sio

import tf_pose.transfer.config as ccon

__all__ = [
            'pred3D'
            'resolve_res',
            'resolve_camera',
            'predict_ar_res',
            'predict_ar_res_wt',
            'select_est'
            ]

def resolve_res(r):
    newr = np.zeros((3, 3))
    newr[:2, 0] = r
    newr[2, 2] = 1
    newr[1::-1, 1] = r
    newr[0, 1] *= -1
    return newr
def resolve_camera(cam):
    new_cam = cam[[0, 2, 1]].copy()
    new_cam = new_cam[:, [0, 2, 1]]
    return new_cam
def predict_ar_res( w, e, s0, camera_r, Lambda, check, a, weights, res, proj_e, residue, Ps, depth_reg, scale_prior):
    frames = w.shape[0]
    points = w.shape[2]
    basis = e.shape[0]
    r = np.empty(2)
    Ps_reshape = Ps.reshape(2 * points)
    w_reshape = w.reshape((frames, points * 2))

    for i in range(check.size):
        c = check[i]
        r[0] = np.cos(c)
        r[1] = np.sin(c)
        grot = camera_r.dot(resolve_res(r))
        rot = grot[:2]
        res[:, :points * 2] = w_reshape
        res[:, :points * 2] -= Ps_reshape
        proj_e[:, :2 * points] = rot.dot(e).transpose(1, 0, 2).reshape(
            e.shape[0], 2 * points)

        if Lambda.size != 0:
            proj_e[:, 2 * points:2 * points + basis] = np.diag(Lambda[:Lambda.shape[0] - 1])
            res[:, 2 * points:].fill(0)
            res[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            # depth regularizer not used
            proj_e[:, 2 * points + basis:] = ((Lambda[Lambda.shape[0] - 1] *
                                               depth_reg) * grot[2]).dot(e)
            # we let the person change scale
            res[:, 2 * points] = scale_prior

        """
        TODO: PLEASE REVIEW THE FOLLOWING CODE....
        overwrite_a and overwrite_b ARE UNEXPECTED ARGUMENTS OF
        scipy.linalg.lstsq
        """
        a[i], residue[i], _, _ = scipy.linalg.lstsq(
            proj_e.T, res.T, overwrite_a=True, overwrite_b=True)

    # find and return best coresponding solution
    best = np.argmin(residue, 0)
    assert (best.shape[0] == frames)
    theta = check[best]
    index = (best, np.arange(frames))
    aa = a.transpose(0, 2, 1)[index]
    retres = residue[index]
    r = np.empty((2, frames))
    r[0] = np.sin(theta)
    r[1] = np.cos(theta)
    return aa, r, retres
def predict_ar_res_wt( w, e, s0, camera_r, Lambda, check, a, weights, res, proj_e, residue, Ps, depth_reg, scale_prior):
    frames = w.shape[0]
    points = w.shape[2]
    basis = e.shape[0]
    r = np.empty(2)
    Ps_reshape = Ps.reshape(2 * points)
    w_reshape = w.reshape((frames, points * 2))
    p_copy = np.empty_like(proj_e)

    for i in range(check.size):
        c = check[i]
        r[0] = np.sin(c)
        r[1] = np.cos(c)
        grot = camera_r.dot(resolve_res(r).T)
        rot = grot[:2]
        rot.dot(s0, Ps)  # TODO: remove?
        res[:, :points * 2] = w_reshape
        res[:, :points * 2] -= Ps_reshape
        proj_e[:, :2 * points] = rot.dot(e).transpose(1, 0, 2).reshape(
            e.shape[0], 2 * points)

        if Lambda.size != 0:
            proj_e[:, 2 * points:2 * points + basis] = np.diag(Lambda[:Lambda.shape[0] - 1])
            res[:, 2 * points:].fill(0)
            res[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, 2 * points + basis:] = ((Lambda[Lambda.shape[0] - 1] *
                                               depth_reg) * grot[2]).dot(e)
            res[:, 2 * points] = scale_prior
        if weights.size != 0:
            res[:, :points * 2] *= weights
        for j in range(frames):
            p_copy[:] = proj_e
            p_copy[:, :points * 2] *= weights[j]
            a[i, :, j], comp_residual, _, _ = np.linalg.lstsq(
                p_copy.T, res[j].T)
            if not comp_residual:
                # equations are over-determined
                residue[i, j] = 1e-5
            else:
                residue[i, j] = comp_residual
    # find and return best coresponding solution
    best = np.argmin(residue, 0)
    index = (best, np.arange(frames))
    theta = check[best]
    aa = a.transpose(0, 2, 1)[index]
    retres = residue[index]
    r = np.empty((2, frames))
    r[0] = np.sin(theta)
    r[1] = np.cos(theta)
    return aa, r, retres
def select_est(w, e, s0, camera_r=None, Lambda=None, weights=None, scale_prior=-0.0014, interval=0.01, depth_reg=0.0325):
    camera_r = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                          ) if camera_r is None else camera_r
    Lambda = np.ones((0, 0)) if Lambda is None else Lambda
    weights = np.ones((0, 0, 0)) if weights is None else weights

    charts = e.shape[0]
    frames = w.shape[0]
    basis = e.shape[1]
    points = e.shape[3]
    assert (s0.shape[0] == charts)
    r = np.empty((charts, 2, frames))
    a = np.empty((charts, frames, e.shape[1]))
    score = np.empty((charts, frames))
    check = np.arange(0, 1, interval) * 2 * np.pi
    cache_a = np.empty((check.size, basis, frames))
    residue = np.empty((check.size, frames))

    if Lambda.size != 0:
        res = np.zeros((frames, points * 2 + basis + points))
        proj_e = np.zeros((basis, 2 * points + basis + points))
    else:
        res = np.empty((frames, points * 2))
        proj_e = np.empty((basis, 2 * points))
    Ps = np.empty((2, points))

    if weights.size == 0:
        for i in range(charts):
            if Lambda.size != 0:
                a[i], r[i], score[i] = predict_ar_res(
                    w, e[i], s0[i], camera_r,
                    Lambda[i], check, cache_a, weights,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
            else:
                a[i], r[i], score[i] = predict_ar_res(
                    w, e[i], s0[i], camera_r, Lambda,
                    check, cache_a, weights,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
    else:
        w2 = weights.reshape(weights.shape[0], -1)
        for i in range(charts):
            if Lambda.size != 0:
                a[i], r[i], score[i] = predict_ar_res_wt(
                    w, e[i], s0[i], camera_r,
                    Lambda[i], check, cache_a, w2,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
            else:
                a[i], r[i], score[i] = predict_ar_res_wt(
                    w, e[i], s0[i], camera_r, Lambda,
                    check, cache_a, w2,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)

    remaining_dims = 3 * w.shape[2] - e.shape[1]
    assert (np.all(score > 0))
    assert (remaining_dims >= 0)
    # Zero problems in log space due to un-regularised first co-efficient
    l = Lambda.copy()
    l[Lambda == 0] = 1
    llambda = -np.log(l)
    score /= 2
    return score, a, r

class pred3D:

    def __init__(self, prob_model_path):
        model_param = sio.loadmat(prob_model_path)
        self.mu = np.reshape(
            model_param['mu'], (model_param['mu'].shape[0], 3, -1))
        self.e = np.reshape(model_param['e'], (model_param['e'].shape[
                            0], model_param['e'].shape[1], 3, -1))
        self.sigma = model_param['sigma']
        self.cam = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    @staticmethod
    def cost3d(model, gt):                                    #finding error in 3D pose in with precisions of mm
        out = np.sqrt(((gt - model) ** 2).sum(1)).mean(-1)
        return out
    @staticmethod
    def renorm_gt(gt):                                        #measure joints and normalise
        _POSE_TREE = np.asarray([
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]]).T
        scale = np.sqrt(((gt[:, :, _POSE_TREE[0]] -
                          gt[:, :, _POSE_TREE[1]]) ** 2).sum(2).sum(1))
        return gt / scale[:, np.newaxis, np.newaxis]
    @staticmethod
    def build_model(a, e, s0):                                #building teh model
        assert (s0.shape[1] == 3)
        assert (e.shape[2] == 3)
        assert (a.shape[1] == e.shape[1])
        out = np.einsum('...i,...ijk', a, e)
        out += s0
        return out
    @staticmethod
    def build_and_rot_model(a, e, s0, r):                     #build and rotate the model
        from numpy.core.umath_tests import matrix_multiply

        r2 = pred3D.resolve_res(r.T).transpose((0, 2, 1))
        mod = pred3D.build_model(a, e, s0)
        mod = matrix_multiply(r2, mod)
        return mod
    @staticmethod
    def resolve_res(r):                                       #converts planar coefficients into rotation matrix
        assert (r.ndim == 2)                                  # Technically optional assert, but if this fails data is probably transposed
        assert (r.shape[1] == 2)
        assert (np.all(np.isfinite(r)))
        norm = np.sqrt((r[:, :2] ** 2).sum(1))
        assert (np.all(norm > 0))
        r /= norm[:, np.newaxis]
        assert (np.all(np.isfinite(r)))
        newr = np.zeros((r.shape[0], 3, 3))
        newr[:, :2, 0] = r[:, :2]
        newr[:, 2, 2] = 1
        newr[:, 1::-1, 1] = r[:, :2]
        newr[:, 0, 1] *= -1
        return newr
    @staticmethod
    def centre(data_2d):                                      #centering the data
        return (data_2d.T - data_2d.mean(1)).T
    @staticmethod
    def centre_all(data):                                     #center all data
        if data.ndim == 2:
            return pred3D.centre(data)
        return (data.transpose(2, 0, 1) - data.mean(2)).transpose(1, 2, 0)
    @staticmethod
    def normalise_data(d2, weights):
        #normalise data as per the height
        # the joints with weight set to 0 should not be considered in the
        # normalisation process
        d2 = d2.reshape(d2.shape[0], -1, 2).transpose(0, 2, 1)
        idx_consider = weights[0, 0].astype(np.bool)
        if np.sum(weights[:, 0].sum(1) >= ccon.MIN_NUM_JOINTS) == 0:
            raise Exception('Not enough 2D joints identified to generate 3D pose')
        d2[:, :, idx_consider] = pred3D.centre_all(d2[:, :, idx_consider])

        # Height normalisation (2 meters)
        m2 = d2[:, 1, idx_consider].min(1) / 2.0
        m2 -= d2[:, 1, idx_consider].max(1) / 2.0
        crap = m2 == 0
        m2[crap] = 1.0
        d2[:, :, idx_consider] /= m2[:, np.newaxis, np.newaxis]
        return d2, m2
    @staticmethod
    def transform_joints(pose_2d, visible_joints):
        _H36M_ORDER = [8, 9, 10, 11, 12, 13, 1, 0, 5, 6, 7, 2, 3, 4]
        _W_POS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        def swap_xy(poses):
            tmp = np.copy(poses[:, :, 0])
            poses[:, :, 0] = poses[:, :, 1]
            poses[:, :, 1] = tmp
            return poses
        assert (pose_2d.ndim == 3)
        new_pose = pose_2d.copy()
        # new_pose = swap_xy(new_pose)      # not used
        new_pose = new_pose[:, _H36M_ORDER]
        # defining weights according to occlusions
        weights = np.zeros((pose_2d.shape[0], 2, ccon.H36M_NUM_JOINTS))
        ordered_visibility = np.repeat(
            visible_joints[:, _H36M_ORDER, np.newaxis], 2, 2
        ).transpose([0, 2, 1])
        weights[:, :, _W_POS] = ordered_visibility
        return new_pose, weights
    def affine_estimate(self, w, depth_reg=0.085, weights=None, scale=10.0,
                        scale_mean=0.0016 * 1.8 * 1.2, scale_std=1.2 * 0,
                        cap_scale=-0.00129):
        """
        Quick switch to allow reconstruction at unknown scale returns a,r
        and scale
        """
        weights = np.zeros((0, 0, 0)) if weights is None else weights

        s = np.empty((self.sigma.shape[0], self.sigma.shape[1] + 4))  # e,y,x,z
        s[:, :4] = 10 ** -5  # Tiny but makes stuff well-posed
        s[:, 0] = scale_std
        s[:, 4:] = self.sigma
        s[:, 4:-1] *= scale

        e2 = np.zeros((self.e.shape[0], self.e.shape[
                      1] + 4, 3, self.e.shape[3]))
        e2[:, 1, 0] = 1.0
        e2[:, 2, 1] = 1.0
        e2[:, 3, 0] = 1.0
        # This makes the least_squares problem ill posed, as X,Z are
        # interchangable
        # Hence regularisation above to speed convergence and stop blow-up
        e2[:, 0] = self.mu
        e2[:, 4:] = self.e
        t_m = np.zeros_like(self.mu)

        res, a, r = select_est(w, e2, t_m, self.cam, s, weights=weights,
                           interval=0.01, depth_reg=depth_reg,
                           scale_prior=scale_mean)

        scale = a[:, :, 0]
        reestimate = scale > cap_scale
        m = self.mu * cap_scale
        for i in range(scale.shape[0]):
            if reestimate[i].sum() > 0:
                ehat = e2[i:i + 1, 1:]
                mhat = m[i:i + 1]
                shat = s[i:i + 1, 1:]
                (res2, a2, r2) = select_est(
                    w[reestimate[i]], ehat, mhat, self.cam, shat,
                    weights=weights[reestimate[i]],
                    interval=0.01, depth_reg=depth_reg,
                    scale_prior=scale_mean
                )
                res[i:i + 1, reestimate[i]] = res2
                a[i:i + 1, reestimate[i], 1:] = a2
                a[i:i + 1, reestimate[i], 0] = cap_scale
                r[i:i + 1, :, reestimate[i]] = r2
        scale = a[:, :, 0]
        a = a[:, :, 1:] / a[:, :, 0][:, :, np.newaxis]
        return res, e2[:, 1:], a, r, scale
    def better_rec(self, w, model, s=1, weights=1, damp_z=1):
        """Quick switch to allow reconstruction at unknown scale
        returns a,r and scale"""
        from numpy.core.umath_tests import matrix_multiply
        proj = matrix_multiply(self.cam[np.newaxis], model)
        proj[:, :2] = (proj[:, :2] * s + w * weights) / (s + weights)
        proj[:, 2] *= damp_z
        out = matrix_multiply(self.cam.T[np.newaxis], proj)
        return out
    def create_rec(self, w2, weights, res_weight=1):
        """Reconstruct 3D pose given a 2D pose"""
        _SIGMA_SCALING = 5.2

        res, e, a, r, scale = self.affine_estimate(
            w2, scale=_SIGMA_SCALING, weights=weights,
            depth_reg=0, cap_scale=-0.001, scale_mean=-0.003
        )

        remaining_dims = 3 * w2.shape[2] - e.shape[1]
        assert (remaining_dims >= 0)
        llambda = -np.log(self.sigma)
        lgdet = np.sum(llambda[:, :-1], 1) + llambda[:, -1] * remaining_dims
        score = (res * res_weight + lgdet[:, np.newaxis] * (scale ** 2))
        best = np.argmin(score, 0)
        index = np.arange(best.shape[0])
        a2 = a[best, index]
        r2 = r[best, :, index].T
        rec = pred3D.build_and_rot_model(a2, e[best], self.mu[best], r2)
        rec *= -np.abs(scale[best, index])[:, np.newaxis, np.newaxis]

        rec = self.better_rec(w2, rec, 1, 1.55 * weights, 1) * -1
        rec = pred3D.renorm_gt(rec)
        rec *= 0.97
        return rec
    def compute_3d(self, pose_2d, weights):                  #Reconstruct 3D poses given 2D estimations
        _J_POS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
        _SCALE_3D = 1174.88312988

        if pose_2d.shape[1] != ccon.H36M_NUM_JOINTS:
            # need to call the linear regressor
            reg_joints = np.zeros(
                (pose_2d.shape[0], ccon.H36M_NUM_JOINTS, 2))
            for oid, singe_pose in enumerate(pose_2d):
                reg_joints[oid, _J_POS] = singe_pose

            norm_pose, _ = pred3D.normalise_data(reg_joints, weights)
        else:
            norm_pose, _ = pred3D.normalise_data(pose_2d, weights)

        pose_3d = self.create_rec(norm_pose, weights) * _SCALE_3D
        return pose_3d
