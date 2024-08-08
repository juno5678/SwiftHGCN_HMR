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

from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, '/home/juno/MambaHMR')
# sys.path.insert(0, '/home/juno/MambaHMR')

import argparse
import os
import os.path as op
import json
import time
import datetime
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
from src.modeling.model.network import MambaHMR
from src.modeling._smpl import SMPL, Mesh
import src.modeling.data.config as cfg
import torchvision.models as models
from src.datasets.build import make_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_gt_reconstruction, visualize_only_mesh_reconstruction
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection
import matplotlib.pyplot as plt
from src.modeling.backbone.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.backbone.hrnet_32 import HigherResolutionNet
from src.modeling.backbone.config import config as hrnet_config
from src.modeling.backbone.config import update_config as hrnet_update_config

from src.utils.image_ops import uncrop, uncrop_paste
from src.tools.loss import *
from tqdm import tqdm
from azureml.core.run import Run

aml_run = Run.get_context()


def save_checkpoint(model, args, optimzier, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            torch.save(optimzier.state_dict(), op.join(checkpoint_dir, 'op_state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split + '_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs / 2.0)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


def run_eval_general(args, val_dataloader, Network, smpl, mesh_sampler, renderer):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    epoch = 0
    if args.distributed:
        Network = torch.nn.parallel.DistributedDataParallel(
            Network, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    Network.eval()

    # val_mPVE, val_mPJPE, val_PAmPJPE, val_refine_mPVE, val_refine_mPJPE, val_refine_PAmPJPE, val_count, val_loss \
    #                                                             = run_validate(args, val_dataloader,
    #                                                                      Network,
    #                                                                      criterion_keypoints,
    #                                                                      criterion_vertices,
    #                                                                      epoch,
    #                                                                      smpl,
    #                                                                      mesh_sampler)
    # val_mPVE, val_mPJPE, val_PAmPJPE, val_count, val_loss \
    #     = run_validate(args, val_dataloader,
    #                    Network,
    #                    criterion_keypoints,
    #                    criterion_vertices,
    #                    epoch,
    #                    smpl,
    #                    mesh_sampler)
    ls_val_mPVE, ls_val_mPJPE, ls_val_PAmPJPE, ms_val_mPVE, ms_val_mPJPE, ms_val_PAmPJPE, hs_val_mPVE, hs_val_mPJPE, hs_val_PAmPJPE, val_count, val_loss = run_validate(
        args, val_dataloader,
        Network,
        criterion_keypoints,
        criterion_vertices,
        epoch,
        smpl,
        mesh_sampler, renderer)

    aml_run.log(name='mPVE', value=float(1000 * hs_val_mPVE))
    aml_run.log(name='mPJPE', value=float(1000 * hs_val_mPJPE))
    aml_run.log(name='PAmPJPE', value=float(1000 * hs_val_PAmPJPE))

    # aml_run.log(name='refine mPVE', value=float(1000 * val_refine_mPVE))
    # aml_run.log(name='refine mPJPE', value=float(1000 * val_refine_mPJPE))
    # aml_run.log(name='refine PAmPJPE', value=float(1000 * val_refine_PAmPJPE))
    #
    logger.info(
        ' '.join(['Validation', 'epoch: {ep}', ]).format(ep=epoch)
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000 * hs_val_mPVE, 1000 * hs_val_mPJPE,
                                                                      1000 * hs_val_PAmPJPE)
    #     + '  refine mPVE: {:6.2f}, refine mPJPE: {:6.2f}, refine PAmPJPE: {:6.2f} '.format(1000 * val_refine_mPVE,
    #                                                                   1000 * val_refine_mPJPE,
    #                                                                   1000 * val_refine_PAmPJPE)
    )
    # checkpoint_dir = save_checkpoint(Network, args, 0, 0)
    return


def cal_acc(gt_vertices, has_smpl, gt_3d_joints, has_3d_joints, pred_vertices, smpl):
    with torch.no_grad():
        pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

        pred_3d_pelvis = pred_3d_joints_from_smpl[:, cfg.H36M_J17_NAME.index('Pelvis'), :]
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
        error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints)
        error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(),
                                               gt_3d_joints[:, :, :3].cpu().numpy(),
                                               reduction=None)
    return error_joints, error_joints_pa, error_vertices

def record_error_sample(img_keys, mpvpe, min, max,  error_file):

    error_idx = np.where(((mpvpe >= min) & (mpvpe < max)))
    be_img = [img_keys[error_idx[0][i]] for i in range(len(error_idx[0]))]
    be_mpvpe = [mpvpe[error_idx[0][i]] for i in range(len(error_idx[0]))]

    with open(error_file, 'a') as f:
        for i in range(len(be_img)):
            f.write(be_img[i] + '\t' + str(be_mpvpe[i]) + '\n')

    return len(be_img), error_idx[0]

def run_validate(args, val_dataloader, Network, criterion_keypoints, criterion_vertices, epoch, smpl, mesh_sampler, renderer=None):
    batch_time = AverageMeter()
    ls_mPVE = AverageMeter()
    ls_mPJPE = AverageMeter()
    ls_PAmPJPE = AverageMeter()

    ms_mPVE = AverageMeter()
    ms_mPJPE = AverageMeter()
    ms_PAmPJPE = AverageMeter()

    hs_mPVE = AverageMeter()
    hs_mPJPE = AverageMeter()
    hs_PAmPJPE = AverageMeter()

    val_loss = AverageMeter()
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)

    l_e_count = 0
    n_e_count = 0
    h_e_count = 0
    hy_e_count = 0
    s_e_count = 0

    l_e_file = "./output/low_error.tsv"
    n_e_file = "./output/normal_error.tsv"
    h_e_file = "./output/high_error.tsv"
    hy_e_file = "./output/hyper_error.tsv"
    s_e_file = "./output/super_error.tsv"
    # switch to evaluate mode
    Network.eval()
    smpl.eval()
    run = False
    next_stop = False
    with torch.no_grad():
        # end = time.time()
        for i, (img_keys, images, annotations) in enumerate(val_dataloader):
            if next_stop == True and run == False:
                break
            if "downtown_rampAndStairs_00/image_00785.jpg" in img_keys:
                run = True
            elif "downtown_rampAndStairs_00/image_00815.jpg" in img_keys:
                next_stop = True
                run = False
            if run == False:
                continue
            print(i, img_keys)
            img_info = {}
            img_info['full_img'] = annotations['full_img']
            img_info['img_key'] = img_keys
            img_info['orig_shape'] = annotations['orig_shape']
            img_info['scale'] = annotations['scale']
            img_info['center'] = annotations['center']
            batch_size = images.size(0)
            # compute output
            images = images.cuda(args.device)
            # gt_3d_joints, has_3d_joints, has_smpl, gt_vertices = get_acc_gt(args, annotations, smpl, mesh_sampler)
            gt_2d_joints, gt_3d_joints, has_3d_joints, has_2d_joints, has_smpl, gt_vertices_sub2, gt_vertices_sub, gt_vertices, meta_lf_masks, meta_mf_masks, meta_hf_masks = get_loss_gt(
                args, annotations, smpl, mesh_sampler)
            # forward-pass
            ls_outputs, ms_outputs, hs_outputs = Network(images)

            ls_pred_camera, ls_pred_3d_joints, ls_pred_vertices_sub2, ls_pred_vertices_sub, ls_pred_vertices = ls_outputs
            ms_pred_camera, ms_pred_3d_joints, ms_pred_vertices_sub2, ms_pred_vertices_sub, ms_pred_vertices = ms_outputs
            hs_pred_camera, hs_pred_3d_joints, hs_pred_vertices_sub2, hs_pred_vertices_sub, hs_pred_vertices = hs_outputs

            ls_pred_2d_joints_from_smpl, ls_loss = cal_val_losses_nr(
                args,
                ls_pred_camera,
                ls_pred_3d_joints,
                ls_pred_vertices_sub2,
                ls_pred_vertices_sub,
                ls_pred_vertices,
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
                smpl)

            ms_pred_2d_joints_from_smpl, ms_loss = cal_val_losses_nr(
                args,
                ms_pred_camera,
                ms_pred_3d_joints,
                ms_pred_vertices_sub2,
                ms_pred_vertices_sub,
                ms_pred_vertices,
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
                smpl)

            hs_pred_2d_joints_from_smpl, hs_loss = cal_val_losses_nr(
                args,
                hs_pred_camera,
                hs_pred_3d_joints,
                hs_pred_vertices_sub2,
                hs_pred_vertices_sub,
                hs_pred_vertices,
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
                smpl)
            # measure accuracy
            ls_error_joints, ls_error_joints_pa, ls_error_vertices = cal_acc(gt_vertices, has_smpl, gt_3d_joints,
                                                                             has_3d_joints,
                                                                             ls_pred_vertices, smpl)

            ms_error_joints, ms_error_joints_pa, ms_error_vertices = cal_acc(gt_vertices, has_smpl, gt_3d_joints,
                                                                             has_3d_joints,
                                                                             ms_pred_vertices, smpl)

            hs_error_joints, hs_error_joints_pa, hs_error_vertices = cal_acc(gt_vertices, has_smpl, gt_3d_joints,
                                                                             has_3d_joints,
                                                                             hs_pred_vertices, smpl)

            l_count, l_e_index = record_error_sample(img_keys, hs_error_vertices, 0, 0.050, l_e_file)
            l_e_count += l_count
            n_count, n_e_index = record_error_sample(img_keys, hs_error_vertices, 0.050, 0.100,  n_e_file)
            n_e_count += n_count
            h_count, h_e_index = record_error_sample(img_keys, hs_error_vertices, 0.100, 0.150,  h_e_file)
            h_e_count += h_count
            hy_count, hy_e_index = record_error_sample(img_keys, hs_error_vertices, 0.150, 0.200,  hy_e_file)
            hy_e_count += hy_count
            s_count, s_e_index = record_error_sample(img_keys, hs_error_vertices, 0.200, 1.000,  s_e_file)
            s_e_count += s_count

            loss = 0.2 * ls_loss + 0.3 * ms_loss + 0.5 * hs_loss

            # measure accuracy
            if len(ls_error_vertices) > 0:
                ls_mPVE.update(np.mean(ls_error_vertices), int(torch.sum(has_smpl)))
            if len(ls_error_joints) > 0:
                ls_mPJPE.update(np.mean(ls_error_joints), int(torch.sum(has_3d_joints)))
            if len(ls_error_joints_pa) > 0:
                ls_PAmPJPE.update(np.mean(ls_error_joints_pa), int(torch.sum(has_3d_joints)))
            # measure accuracy
            if len(ms_error_vertices) > 0:
                ms_mPVE.update(np.mean(ms_error_vertices), int(torch.sum(has_smpl)))
            if len(ms_error_joints) > 0:
                ms_mPJPE.update(np.mean(ms_error_joints), int(torch.sum(has_3d_joints)))
            if len(ms_error_joints_pa) > 0:
                ms_PAmPJPE.update(np.mean(ms_error_joints_pa), int(torch.sum(has_3d_joints)))
            # measure accuracy
            if len(hs_error_vertices) > 0:
                hs_mPVE.update(np.mean(hs_error_vertices), int(torch.sum(has_smpl)))
            if len(hs_error_joints) > 0:
                hs_mPJPE.update(np.mean(hs_error_joints), int(torch.sum(has_3d_joints)))
            if len(hs_error_joints_pa) > 0:
                hs_PAmPJPE.update(np.mean(hs_error_joints_pa), int(torch.sum(has_3d_joints)))

            # if len(loss) > 0:
            val_loss.update(loss.item(), batch_size)
            # hs_visual_imgs = visualize_each_mesh(args,
            #                                 img_info,
            #                                 renderer,
            #                                 annotations['ori_img'].detach(),
            #                                 annotations['joints_2d'].detach(),
            #                                 hs_pred_vertices.detach(),
            #                                 hs_pred_camera.detach(),
            #                                 hs_pred_2d_joints_from_smpl.detach())
            # ls_visual_imgs = visualize_each_mesh(args,
            #                                      img_info,
            #                                      renderer,
            #                                      annotations['ori_img'].detach(),
            #                                      annotations['joints_2d'].detach(),
            #                                      gt_vertices.detach(),
            #                                      ls_pred_vertices.detach(),
            #                                      ls_pred_camera.detach(),
            #                                      ls_pred_2d_joints_from_smpl.detach(), "low")
            # ms_visual_imgs = visualize_each_mesh(args,
            #                                      img_info,
            #                                      renderer,
            #                                      annotations['ori_img'].detach(),
            #                                      annotations['joints_2d'].detach(),
            #                                      gt_vertices.detach(),
            #                                      ms_pred_vertices.detach(),
            #                                      ms_pred_camera.detach(),
            #                                      hs_pred_2d_joints_from_smpl.detach(), "middle")
            hs_visual_imgs = visualize_each_mesh(args,
                                                 img_info,
                                                 renderer,
                                                 annotations['ori_img'].detach(),
                                                 annotations['joints_2d'].detach(),
                                                 gt_vertices.detach(),
                                                 hs_pred_vertices.detach(),
                                                 hs_pred_camera.detach(),
                                                 hs_pred_2d_joints_from_smpl.detach(), "high")

            # if h_count > 0:
            #     hs_visual_imgs = visualize_spec_mesh(renderer,
            #                                     annotations['ori_img'].detach(),
            #                                     annotations['joints_2d'].detach(),
            #                                     hs_pred_vertices.detach(),
            #                                     gt_vertices.detach(),
            #                                     hs_pred_camera.detach(),
            #                                     hs_pred_2d_joints_from_smpl.detach(),
            #                                     h_e_index)
            #     hs_visual_imgs = hs_visual_imgs.transpose(0, 1)
            #     hs_visual_imgs = hs_visual_imgs.transpose(1, 2)
            #     hs_visual_imgs = np.asarray(hs_visual_imgs)
            #
            #     if is_main_process() == True:
            #
            #         hs_img_dir = op.join(args.output_dir, "high_error_val_img")
            #         try:
            #             mkdir(hs_img_dir)
            #         except FileExistsError:
            #             pass
            #         stamp = str(epoch) + '_' + str(i)
            #         output_img = np.asarray(hs_visual_imgs[:, :, ::-1] * 255).astype(np.uint8).copy()
            #         for i in range(len(h_e_index)):
            #             position = (395, 40 + 225 * i)
            #             cv2.putText(output_img, str(int(hs_error_vertices[h_e_index[i]]*1000)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #         temp_fname = hs_img_dir + '/high_error_val_visual_' + stamp + '.jpg'
            #         cv2.imwrite(temp_fname, output_img)
            #         aml_run.log_image(name='high_error visual results', path=temp_fname)
            # if n_count > 0:
            #     hs_visual_imgs = visualize_spec_mesh(renderer,
            #                                          annotations['ori_img'].detach(),
            #                                          annotations['joints_2d'].detach(),
            #                                          hs_pred_vertices.detach(),
            #                                          gt_vertices.detach(),
            #                                          hs_pred_camera.detach(),
            #                                          hs_pred_2d_joints_from_smpl.detach(),
            #                                          n_e_index)
            #     hs_visual_imgs = hs_visual_imgs.transpose(0, 1)
            #     hs_visual_imgs = hs_visual_imgs.transpose(1, 2)
            #     hs_visual_imgs = np.asarray(hs_visual_imgs)
            #
            #     if is_main_process() == True:
            #
            #         hs_img_dir = op.join(args.output_dir, "normal_error_val_img")
            #         try:
            #             mkdir(hs_img_dir)
            #         except FileExistsError:
            #             pass
            #         stamp = str(epoch) + '_' + str(i)
            #         output_img = np.asarray(hs_visual_imgs[:, :, ::-1] * 255).astype(np.uint8).copy()
            #         for i in range(len(n_e_index)):
            #             position = (395, 40 + 225 * i)
            #             cv2.putText(output_img, str(int(hs_error_vertices[n_e_index[i]] * 1000)), position,
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #         temp_fname = hs_img_dir + '/normal_error_val_visual_' + stamp + '.jpg'
            #         cv2.imwrite(temp_fname, output_img)
            #         aml_run.log_image(name='normal_error visual results', path=temp_fname)

            # visualize
            # if s_count > 0:
            #     hs_visual_imgs = visualize_spec_mesh(renderer,
            #                                     annotations['ori_img'].detach(),
            #                                     annotations['joints_2d'].detach(),
            #                                     hs_pred_vertices.detach(),
            #                                     gt_vertices.detach(),
            #                                     hs_pred_camera.detach(),
            #                                     hs_pred_2d_joints_from_smpl.detach(),
            #                                     s_e_index)
            #     hs_visual_imgs = hs_visual_imgs.transpose(0, 1)
            #     hs_visual_imgs = hs_visual_imgs.transpose(1, 2)
            #     hs_visual_imgs = np.asarray(hs_visual_imgs)
            #
            #     if is_main_process() == True:
            #
            #         hs_img_dir = op.join(args.output_dir, "super_error_val_img")
            #         try:
            #             mkdir(hs_img_dir)
            #         except FileExistsError:
            #             pass
            #         stamp = str(epoch) + '_' + str(i)
            #         output_img = np.asarray(hs_visual_imgs[:, :, ::-1] * 255).astype(np.uint8).copy()
            #         for i in range(len(s_e_index)):
            #             position = (395, 40 + 225 * i)
            #             cv2.putText(output_img, str(int(hs_error_vertices[s_e_index[i]]*1000)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #         temp_fname = hs_img_dir + '/super_error_val_visual_' + stamp + '.jpg'
            #         cv2.imwrite(temp_fname, output_img)
            #         aml_run.log_image(name='super_error visual results', path=temp_fname)
            #
            # if l_count > 0:
            #     hs_visual_imgs = visualize_spec_mesh(renderer,
            #                                     annotations['ori_img'].detach(),
            #                                     annotations['joints_2d'].detach(),
            #                                     hs_pred_vertices.detach(),
            #                                     gt_vertices.detach(),
            #                                     hs_pred_camera.detach(),
            #                                     hs_pred_2d_joints_from_smpl.detach(),
            #                                     l_e_index)
            #     hs_visual_imgs = hs_visual_imgs.transpose(0, 1)
            #     hs_visual_imgs = hs_visual_imgs.transpose(1, 2)
            #     hs_visual_imgs = np.asarray(hs_visual_imgs)
            #
            #     if is_main_process() == True:
            #
            #         hs_img_dir = op.join(args.output_dir, "lowerror_val_img")
            #         try:
            #             mkdir(hs_img_dir)
            #         except FileExistsError:
            #             pass
            #         stamp = str(epoch) + '_' + str(i)
            #         output_img = np.asarray(hs_visual_imgs[:, :, ::-1] * 255).astype(np.uint8).copy()
            #         for i in range(len(l_e_index)):
            #             position = (395, 40 + 225 * i)
            #             cv2.putText(output_img, str(int(hs_error_vertices[l_e_index[i]]*1000)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            #         temp_fname = hs_img_dir + '/low_error_val_visual_' + stamp + '.jpg'
            #         cv2.imwrite(temp_fname, output_img)
            #         aml_run.log_image(name='low_error visual results', path=temp_fname)



    print("low error count : ", l_e_count)
    print("normal error count : ", n_e_count)
    print("high error count : ", h_e_count)
    print("hyper error count : ", hy_e_count)
    print("super error count : ", s_e_count)
    ls_val_mPVE = all_gather(float(ls_mPVE.avg))
    ls_val_mPVE = sum(ls_val_mPVE) / len(ls_val_mPVE)
    ls_val_mPJPE = all_gather(float(ls_mPJPE.avg))
    ls_val_mPJPE = sum(ls_val_mPJPE) / len(ls_val_mPJPE)
    ls_val_PAmPJPE = all_gather(float(ls_PAmPJPE.avg))
    ls_val_PAmPJPE = sum(ls_val_PAmPJPE) / len(ls_val_PAmPJPE)

    ms_val_mPVE = all_gather(float(ms_mPVE.avg))
    ms_val_mPVE = sum(ms_val_mPVE) / len(ms_val_mPVE)
    ms_val_mPJPE = all_gather(float(ms_mPJPE.avg))
    ms_val_mPJPE = sum(ms_val_mPJPE) / len(ms_val_mPJPE)
    ms_val_PAmPJPE = all_gather(float(ms_PAmPJPE.avg))
    ms_val_PAmPJPE = sum(ms_val_PAmPJPE) / len(ms_val_PAmPJPE)

    hs_val_mPVE = all_gather(float(hs_mPVE.avg))
    hs_val_mPVE = sum(hs_val_mPVE) / len(hs_val_mPVE)
    hs_val_mPJPE = all_gather(float(hs_mPJPE.avg))
    hs_val_mPJPE = sum(hs_val_mPJPE) / len(hs_val_mPJPE)
    hs_val_PAmPJPE = all_gather(float(hs_PAmPJPE.avg))
    hs_val_PAmPJPE = sum(hs_val_PAmPJPE) / len(hs_val_PAmPJPE)

    val_count = all_gather(float(hs_mPVE.count))
    val_count = sum(val_count)

    val_loss = val_loss.avg

    return ls_val_mPVE, ls_val_mPJPE, ls_val_PAmPJPE,ms_val_mPVE, ms_val_mPJPE, ms_val_PAmPJPE, hs_val_mPVE, hs_val_mPJPE, hs_val_PAmPJPE, val_count, val_loss

def visualize_each_mesh(args,
                   img_info,
                   renderer,
                   images,
                   gt_keypoints_2d,
                   gt_vertices,
                   pred_vertices,
                   pred_camera,
                   pred_keypoints_2d,
                   level):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # flip = 0  # flipping
    # pn = np.ones(3)  # per channel pixel-noise
    # rot = 0  # rotation
    # sc = 1  # scaling
    # Do visualization for the first 6 images of the batch
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        gt_vertice = gt_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        # gt_rend_img = visualize_only_mesh_reconstruction(img, 224, gt_keypoints_2d_, gt_vertice, pred_keypoints_2d_, cam,
        #                                               renderer)
        rend_img, skeleton_img = visualize_only_mesh_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        # rend_img = rend_img.transpose(2, 0, 1)
        # rend_img = torch.from_numpy(rend_img)
        # img, center, scale, orig_shape,
        orig_img = img_info['full_img'][i][:, :, [2, 1, 0]]

        orig_shape = [img_info['orig_shape'][1][i], img_info['orig_shape'][0][i], 3]
        full_img = uncrop_paste(np.asarray(rend_img[:, :, ::-1] * 255), img_info['center'][i], img_info['scale'][i], orig_shape, orig_img)
        # gt_full_img = uncrop_paste(np.asarray(gt_rend_img[:, :, ::-1] * 255), img_info['center'][i], img_info['scale'][i],
        #                         orig_shape, orig_img)
        img_dir = op.join(args.output_dir, "hs_val_img")
        hs_img_dir = op.join(img_dir, level)
        mesh_img_dir = op.join(hs_img_dir, "mesh")
        skeleton_img_dir = op.join(hs_img_dir, "skeleton")
        orig_img_dir = op.join(hs_img_dir, "orig")
        full_img_dir = op.join(hs_img_dir, "full")
        gt_full_img_dir = op.join(hs_img_dir, "gt_full")
        gt_mesh_img_dir = op.join(hs_img_dir, "gt_mesh")
        try:
            mkdir(hs_img_dir)
            mkdir(mesh_img_dir)
            mkdir(skeleton_img_dir)
            mkdir(orig_img_dir)
            mkdir(full_img_dir)
            mkdir(gt_full_img_dir)
            mkdir(gt_mesh_img_dir)
        except FileExistsError:
            pass

        filename = img_info['img_key'][i].replace('/', '_')
        temp_fname = mesh_img_dir + '/' +  filename
        full_fname = full_img_dir + '/' + filename
        skeleton_fname = skeleton_img_dir + '/' + filename
        gt_full_fname = gt_full_img_dir + '/' + filename
        gt_mesh_fname = gt_mesh_img_dir + '/' + filename
        orig_fname = orig_img_dir + '/' + filename

        cv2.imwrite(temp_fname, np.asarray(rend_img[:, :, ::-1] * 255))
        cv2.imwrite(skeleton_fname, np.asarray(skeleton_img[:, :, ::-1] * 255))
        # cv2.imwrite(gt_mesh_fname, np.asarray(gt_rend_img[:, :, ::-1] * 255))
        cv2.imwrite(full_fname, full_img)
        # cv2.imwrite(gt_full_fname, gt_full_img)
        cv2.imwrite(orig_fname, np.asarray(orig_img))
        aml_run.log_image(name='hs visual results', path=temp_fname)
    return rend_imgs
def visualize_mesh(renderer,
                   images,
                   gt_keypoints_2d,
                   pred_vertices,
                   pred_camera,
                   pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 6)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2, 0, 1)
        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs
def visualize_spec_mesh(renderer,
                   images,
                   gt_keypoints_2d,
                   pred_vertices,
                   gt_vertices,
                   pred_camera,
                   pred_keypoints_2d,
                   index):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    # for i in range(min(batch_size, 10)):
    for i in index:
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        gt_vertice = gt_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        # rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = visualize_gt_reconstruction(img, 224, gt_keypoints_2d_, vertices, gt_vertice, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2, 0, 1)
        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_test(renderer,
                        images,
                        gt_keypoints_2d,
                        pred_vertices,
                        pred_camera,
                        pred_keypoints_2d,
                        PAmPJPE_h36m_j14):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE_h36m_j14[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam,
                                                 renderer, score)
        rend_img = rend_img.transpose(2, 0, 1)
        rend_imgs.append(torch.from_numpy(rend_img))
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    # parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
    #                     help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_op_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float,
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=50.0, type=float)
    parser.add_argument("--refine_vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--refine_joint_loss_weight", default=1000.0, type=float)
    parser.add_argument("--joints_loss_weight", default=500.0, type=float)
    parser.add_argument("--edge_loss_weight", default=10.0, type=float)
    parser.add_argument("--normal_loss_weight", default=10.0, type=float)
    parser.add_argument("--heatmap_loss_weight", default=10.0, type=float)

    parser.add_argument("--vloss_w_full", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--transformer_nhead", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                             "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--model_dim", default=512, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--feedforward_dim_1", default=1024, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--feedforward_dim_2", default=512, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--position_dim", default=128, type=int,
                        help="position dim.")
    parser.add_argument("--activation", default="relu", type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="The Image Feature Dimension.")
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand")
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true', )
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88,
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    #########################################################
    # Vim
    #########################################################
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        # print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()

    # Renderer for visualization
    renderer = Renderer(faces=smpl.faces.cpu().numpy())

    # if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
    #     # if only run eval, load checkpoint
    #     logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
    #     _model = torch.load(args.resume_checkpoint)
    # else:
    # init ImageNet pre-trained backbone model
    if args.arch == 'hrnet-w40':
        hrnet_yaml = './models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = './models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch == 'hrnet-w64':
        hrnet_yaml = './models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = './models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w64 model')
    elif args.arch == 'hrnet-w32':
        backbone = HigherResolutionNet(args)
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    if args.run_eval_only == True and args.resume_checkpoint != None and args.resume_checkpoint != 'None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _model = torch.load(args.resume_checkpoint)
    else:
        _model = MambaHMR(args, mesh_sampler, backbone)

        if args.resume_checkpoint != None and args.resume_checkpoint != 'None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            # workaround approach to load sparse tensor in graph conv.
            states = torch.load(args.resume_checkpoint, map_location=args.device)

            for k, v in states.items():
                states[k] = v.cpu()
            _model.load_state_dict(states, strict=False)

            del states
            gc.collect()
            torch.cuda.empty_cache()

    _model.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only == True:
        val_dataloader = make_data_loader(args, args.val_yaml,
                                          args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_general(args, val_dataloader, _model, smpl, mesh_sampler, renderer)



if __name__ == "__main__":
    args = parse_args()
    main(args)
