"""
----------------------------------------------------------------------------------------------
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
Modified from MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

import torch
import src.modeling.data.config as cfg
from src.utils.geometric_layers import orthographic_projection
from torch.nn import functional as F
import math

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error


def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def make_gt(verts_camed, has_smpl, MAP, img_size=112):
    verts_camed = ((verts_camed[has_smpl==1] + 1) * 0.5) * img_size
    x = verts_camed[:, :,0].long()
    y = verts_camed[:, :,1].long()

    indx = img_size*y + x
    flag1 = indx<img_size*img_size
    flag2 = -1 < indx
    flag = flag2*flag1
    MAP = MAP.to(verts_camed.device)
    GT = MAP[indx[flag]].reshape(-1,1,img_size,img_size).to(verts_camed.device)
    # GT = conv_gauss(GT, device=verts_camed.device)
    #
    # GT[GT==0] = -0.1

    return GT, flag

def conv_gauss(img,device):
    k = torch.Tensor([[ 1.25, 2.5, 1.25]])
    kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1).to(device)
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel)


def get_NormalVectorLoss(coord_out, coord_gt, face, has_smpl):
    # face = torch.LongTensor(face).cuda()
    epsilon = 1e-8

    # 过滤有效的样本
    coord_out = coord_out[has_smpl == 1]
    coord_gt = coord_gt[has_smpl == 1]

    # 计算每个面的边向量并归一化
    v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
    v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
    v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]

    v1_out = F.normalize(v1_out, p=2, dim=2, eps=epsilon)  # L2 归一化
    v2_out = F.normalize(v2_out, p=2, dim=2, eps=epsilon)  # L2 归一化
    v3_out = F.normalize(v3_out, p=2, dim=2, eps=epsilon)  # L2 归一化

    v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
    v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]

    v1_gt = F.normalize(v1_gt, p=2, dim=2, eps=epsilon)  # L2 归一化
    v2_gt = F.normalize(v2_gt, p=2, dim=2, eps=epsilon)  # L2 归一化

    # 计算法向量并归一化
    normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
    normal_gt = F.normalize(normal_gt, p=2, dim=2, eps=epsilon)  # L2 归一化

    # 计算每个边向量与法向量的余弦相似度
    cos1 = torch.abs(torch.sum(v1_out * normal_gt, dim=2, keepdim=True))
    cos2 = torch.abs(torch.sum(v2_out * normal_gt, dim=2, keepdim=True))
    cos3 = torch.abs(torch.sum(v3_out * normal_gt, dim=2, keepdim=True))

    # 合并并计算平均损失
    loss = torch.cat((cos1, cos2, cos3), dim=1)
    loss_mean = loss.mean()
    if math.isnan(loss_mean):
        return 0
    else:
        return loss_mean



def get_EdgeLengthLoss(coord_out, coord_gt, face,has_smpl):
    # face = torch.LongTensor(face).cuda()
    # epsilon = 1e-8
    # 过滤有效的样本
    coord_out = coord_out[has_smpl == 1]
    coord_gt = coord_gt[has_smpl == 1]

    # 计算边长
    d1_out = torch.norm(coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :], dim=2, keepdim=True)
    d2_out = torch.norm(coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :], dim=2, keepdim=True)
    d3_out = torch.norm(coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :], dim=2, keepdim=True)

    d1_gt = torch.norm(coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :], dim=2, keepdim=True)
    d2_gt = torch.norm(coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :], dim=2, keepdim=True)
    d3_gt = torch.norm(coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :], dim=2, keepdim=True)

    # 计算边长差异
    diff1 = torch.abs(d1_out - d1_gt)
    diff2 = torch.abs(d2_out - d2_gt)
    diff3 = torch.abs(d3_out - d3_gt)

    # 拼接所有差异并计算平均损失
    loss = torch.cat((diff1, diff2, diff3), 1)
    loss_mean = loss.mean()
    if math.isnan(loss_mean):
        return 0
    else:
        return loss_mean

def cal_losses(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    heatmap_loss = 0


    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)
    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints + args.heatmap_loss_weight * heatmap_loss
           # + args.vertices_loss_weight * loss_surface

    return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss

def cal_losses_nr(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)


    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))

    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)

    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

    face = smpl.faces
    loss_edge = get_EdgeLengthLoss(pred_vertices, gt_vertices, face, has_smpl)
    loss_normal = get_NormalVectorLoss(pred_vertices, gt_vertices, face, has_smpl)
    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints \
           + args.edge_loss_weight * loss_edge \
           + args.normal_loss_weight * loss_normal

        #  + args.edge_loss_weight * loss_edge \
          # + args.normal_loss_weight * loss_normal

    return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss
def cal_losses_nrj(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                pred_refine_vertices_sub2,
                pred_refine_vertices_sub,
                pred_refine_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_refine_3d_joints_from_smpl = smpl.get_h36m_joints(pred_refine_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]
    pred_refine_3d_joints_from_smpl = pred_refine_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_refine_2d_joints_from_smpl = orthographic_projection(pred_refine_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)


    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))

    refine_loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_refine_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_refine_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_refine_vertices, gt_vertices, has_smpl,
                                                       args.device))
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)
    refine_loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_refine_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)

    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    refine_loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_refine_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints
    refine_loss_3d_joints =  refine_loss_reg_3d_joints

    face = smpl.faces
    loss_edge = get_EdgeLengthLoss(pred_vertices, gt_vertices, face, has_smpl)
    loss_normal = get_NormalVectorLoss(pred_vertices, gt_vertices, face, has_smpl)
    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints \
           + args.refine_joint_loss_weight * refine_loss_3d_joints \
           + args.refine_vertices_loss_weight * refine_loss_vertices \
           + args.refine_vertices_loss_weight * refine_loss_2d_joints\
           + args.edge_loss_weight * loss_edge \
           + args.normal_loss_weight * loss_normal

        #  + args.edge_loss_weight * loss_edge \
          # + args.normal_loss_weight * loss_normal

    return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss

def get_loss_gt(args, annotations, smpl, mesh_sampler):
    gt_2d_joints = annotations['joints_2d'].cuda(args.device)
    gt_2d_joints = gt_2d_joints[:, cfg.J24_TO_J14, :]
    has_2d_joints = annotations['has_2d_joints'].cuda(args.device)

    gt_3d_joints = annotations['joints_3d'].cuda(args.device)
    gt_3d_pelvis = gt_3d_joints[:, cfg.J24_NAME.index('Pelvis'), :3]
    gt_3d_joints = gt_3d_joints[:, cfg.J24_TO_J14, :]
    gt_3d_joints[:, :, :3] = gt_3d_joints[:, :, :3] - gt_3d_pelvis[:, None, :]
    has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

    gt_pose = annotations['pose'].cuda(args.device)
    gt_betas = annotations['betas'].cuda(args.device)
    has_smpl = annotations['has_smpl'].cuda(args.device)

    # generate simplified mesh
    gt_vertices = smpl(gt_pose, gt_betas)
    gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
    gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

    # normalize gt based on smpl's pelvis
    gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
    gt_smpl_3d_pelvis = gt_smpl_3d_joints[:, cfg.H36M_J17_NAME.index('Pelvis'), :]
    gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :]
    gt_vertices_sub = gt_vertices_sub - gt_smpl_3d_pelvis[:, None, :]
    gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

    mjm_mask = annotations['mjm_mask'].cuda(args.device)
    mvm_mask = annotations['mvm_mask'].cuda(args.device)
    mlfm_mask = annotations['mlfm_mask'].cuda(args.device)
    mmfm_mask = annotations['mmfm_mask'].cuda(args.device)
    mhfm_mask = annotations['mhfm_mask'].cuda(args.device)
    # prepare masks for mask vertex/joint modeling
    l_mjm_mask_ = mjm_mask.expand(-1, -1, args.model_dim)
    l_mvm_mask_ = mvm_mask.expand(-1, -1, args.model_dim)
    l_mfm_mask_ = mlfm_mask.expand(-1, -1, args.model_dim)

    m_mjm_mask_ = mjm_mask.expand(-1, -1, args.model_dim//2)
    m_mvm_mask_ = mvm_mask.expand(-1, -1, args.model_dim//2)
    m_mfm_mask_ = mmfm_mask.expand(-1, -1, args.model_dim//2)

    h_mjm_mask_ = mjm_mask.expand(-1, -1, args.model_dim // 4)
    h_mvm_mask_ = mvm_mask.expand(-1, -1, args.model_dim // 4)
    h_mfm_mask_ = mhfm_mask.expand(-1, -1, args.model_dim// 4)
    meta_lf_masks = torch.cat([l_mfm_mask_, l_mjm_mask_, l_mvm_mask_], dim=1)
    meta_mf_masks = torch.cat([m_mfm_mask_, m_mjm_mask_, m_mvm_mask_], dim=1)
    meta_hf_masks = torch.cat([h_mfm_mask_, h_mjm_mask_, h_mvm_mask_], dim=1)

    return (gt_2d_joints, gt_3d_joints, has_3d_joints, has_2d_joints, has_smpl, gt_vertices_sub2, gt_vertices_sub, gt_vertices
            , meta_lf_masks, meta_mf_masks, meta_hf_masks)
def get_acc_gt(args, annotations, smpl, mesh_sampler):
    # generate gt 3d joints
    gt_3d_joints = annotations['joints_3d'].cuda(args.device)
    gt_3d_pelvis = gt_3d_joints[:, cfg.J24_NAME.index('Pelvis'), :3]
    gt_3d_joints = gt_3d_joints[:, cfg.J24_TO_J14, :]
    gt_3d_joints[:, :, :3] = gt_3d_joints[:, :, :3] - gt_3d_pelvis[:, None, :]
    has_3d_joints = annotations['has_3d_joints'].cuda(args.device)
    # generate gt smpl
    gt_pose = annotations['pose'].cuda(args.device)
    gt_betas = annotations['betas'].cuda(args.device)
    has_smpl = annotations['has_smpl'].cuda(args.device)

    # generate simplified mesh
    gt_vertices = smpl(gt_pose, gt_betas)

    # normalize gt based on smpl pelvis
    gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
    gt_smpl_3d_pelvis = gt_smpl_3d_joints[:, cfg.H36M_J17_NAME.index('Pelvis'), :]
    gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]
    return gt_3d_joints, has_3d_joints, has_smpl,gt_vertices


def cal_val_losses_nr(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))

    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)

    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

    face = smpl.faces
    loss_edge = get_EdgeLengthLoss(pred_vertices, gt_vertices, face, has_smpl)
    loss_normal = get_NormalVectorLoss(pred_vertices, gt_vertices, face, has_smpl)

    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints \
           + args.edge_loss_weight * loss_edge \
           + args.normal_loss_weight * loss_normal


    return pred_2d_joints_from_smpl, loss

def cal_val_losses(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                pred_refine_3d_joints,
                pred_refine_vertices_sub2,
                pred_refine_vertices_sub,
                pred_refine_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_refine_3d_joints_from_smpl = smpl.get_h36m_joints(pred_refine_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]
    pred_refine_3d_joints_from_smpl = pred_refine_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_refine_2d_joints_from_smpl = orthographic_projection(pred_refine_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)
    pred_refine_2d_joints = orthographic_projection(pred_refine_3d_joints, pred_camera)



    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    refine_loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_refine_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))

    refine_loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_refine_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_refine_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_refine_vertices, gt_vertices, has_smpl,
                                                       args.device))
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)
    refine_loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_refine_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)

    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    refine_loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_refine_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_refine_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints
    refine_loss_3d_joints = refine_loss_3d_joints + refine_loss_reg_3d_joints

    face = smpl.faces
    loss_edge = get_EdgeLengthLoss(pred_vertices, gt_vertices, face, has_smpl)
    loss_normal = get_NormalVectorLoss(pred_vertices, gt_vertices, face, has_smpl)

    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints \
           + args.refine_joint_loss_weight * refine_loss_3d_joints \
           + args.refine_vertices_loss_weight * refine_loss_vertices \
           + args.refine_vertices_loss_weight * refine_loss_2d_joints \
           + args.edge_loss_weight * loss_edge \
           + args.normal_loss_weight * loss_normal


    return loss

def cal_val_losses_nrj(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                pred_refine_vertices_sub2,
                pred_refine_vertices_sub,
                pred_refine_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_3d_joints,
                has_2d_joints,
                has_smpl,
                criterion_keypoints,
                criterion_2d_keypoints,
                criterion_vertices,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
    pred_refine_3d_joints_from_smpl = smpl.get_h36m_joints(pred_refine_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]
    pred_refine_3d_joints_from_smpl = pred_refine_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    pred_refine_2d_joints_from_smpl = orthographic_projection(pred_refine_3d_joints_from_smpl, pred_camera)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)



    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints, has_3d_joints,
                                      args.device)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl,
                                                       args.device))

    refine_loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_refine_vertices_sub2, gt_vertices_sub2,
                                                       has_smpl, args.device) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_refine_vertices_sub, gt_vertices_sub,
                                                      has_smpl, args.device) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_refine_vertices, gt_vertices, has_smpl,
                                                       args.device))
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)
    refine_loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_refine_3d_joints_from_smpl, gt_3d_joints,
                                          has_3d_joints, args.device)

    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints, has_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    refine_loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_refine_2d_joints_from_smpl, gt_2d_joints,
                                      has_2d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints
    refine_loss_3d_joints = refine_loss_reg_3d_joints

    face = smpl.faces
    loss_edge = get_EdgeLengthLoss(pred_vertices, gt_vertices, face, has_smpl)
    loss_normal = get_NormalVectorLoss(pred_vertices, gt_vertices, face, has_smpl)

    # loss_surface = surface_loss(criterion_surface, pred_vertices, gt_vertices, has_smpl, args.device)
    # print(heatmap_loss)
    # we empirically use hyperparameters to balance difference losses
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints \
           + args.refine_joint_loss_weight * refine_loss_3d_joints \
           + args.refine_vertices_loss_weight * refine_loss_vertices \
           + args.refine_vertices_loss_weight * refine_loss_2d_joints\
           + args.edge_loss_weight * loss_edge \
           + args.normal_loss_weight * loss_normal


    return loss