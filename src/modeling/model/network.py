"""
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.modeling.backbone.hrnet_32 import HigherResolutionNet
from src.modeling.backbone.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.model.regressor import VimMeshRegressor, Vimfeat
from src.modeling.model.regressor import HgcnMeshRegressor, GcnMeshRegressor
from src.modeling.model.module.basic_modules import BasicBlock
from src.modeling.model.module.position_encoding import build_position_encoding
from src.modeling.vim.models_mamba import feat_Mamba, VisionMamba, PatchEmbed

import torchvision.transforms as transforms
BN_MOMENTUM = 0.1


class MambaHMR(torch.nn.Module):
    def __init__(self, args, mesh_sampler, backbone):
        super(MambaHMR, self).__init__()

        self.backbone = backbone  # 128, 56, 56
        self.num_joints = 14
        self.num_vertices = 431
        self.build_head(args)
        self.hs_pos_embedding = build_position_encoding(hidden_dim=args.model_dim//4)
        self.ms_pos_embedding = build_position_encoding(hidden_dim=args.model_dim//2)
        self.ls_pos_embedding = build_position_encoding(hidden_dim=args.model_dim)
        self.ls_fuse_pos_embed = nn.Parameter(torch.zeros(1, 49  + self.num_joints+self.num_vertices+1, args.model_dim))
        self.ms_fuse_pos_embed = nn.Parameter(torch.zeros(1, 196  + self.num_joints+self.num_vertices+1, args.model_dim//2))
        self.hs_fuse_pos_embed = nn.Parameter(torch.zeros(1, 196  + self.num_joints+self.num_vertices+1, args.model_dim//4))
        self.ls_pos_drop = nn.Dropout(p=args.drop_rate)
        self.ms_pos_drop = nn.Dropout(p=args.drop_rate)
        self.hs_pos_drop = nn.Dropout(p=args.drop_rate)
        self.cam_token_embed = nn.Embedding(1, args.model_dim)
        self.joint_token_embed = nn.Embedding(self.num_joints, args.model_dim)
        self.vertex_token_embed = nn.Embedding(self.num_vertices, args.model_dim)

        self.token_position_embed = nn.Embedding(1+49+self.num_joints+self.num_vertices, args.position_dim)

        self.vim_ls_feat = feat_Mamba(embed_dim=384, depth=4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                              final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False,
                              bimamba_type="v2", if_cls_token=False, if_devide_out=True, use_middle_cls_token=True)
        # self.vim_hs_feat = feat_Mamba(embed_dim=192, depth=4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        #                               final_pool_type='mean', if_abs_pos_embed=True, if_rope=False,
        #                               if_rope_residual=False,
        #                               bimamba_type="v2", if_cls_token=False, if_devide_out=True,
        #                               use_middle_cls_token=True)
        self.hs_patch_embed = PatchEmbed(img_size=56, patch_size=4, stride=4, in_chans=args.model_dim//4, embed_dim=args.model_dim//4)
        self.ms_patch_embed = PatchEmbed(img_size=28, patch_size=2, stride=2, in_chans=args.model_dim//2, embed_dim=args.model_dim//2)



        self.fc1 = nn.Linear(3, 192)
        self.fc2 = nn.Linear(3, 96)
        # self.vim_block1 = Vimfeat(args, mesh_sampler, model_dim=(args.model_dim, args.model_dim))
        self.vim_block1 = VimMeshRegressor(args, mesh_sampler, model_dim=(args.model_dim, args.model_dim))
        # self.vim_block2 = Vimfeat(args, mesh_sampler, model_dim=(args.model_dim//2, args.model_dim//2))
        self.vim_block2 = VimMeshRegressor(args, mesh_sampler, model_dim=(args.model_dim // 2, args.model_dim // 2))
        self.vim_block3 = VimMeshRegressor(args, mesh_sampler, model_dim=(args.model_dim//4, args.model_dim//4))
        self.hgcn_block = HgcnMeshRegressor(args, mesh_sampler)
        self.gcn_block = GcnMeshRegressor(args, mesh_sampler)

    def build_head(self, args):


        self.inplanes = self.backbone.backbone_channels
        self.feature_extract_layer = self._make_resnet_layer(BasicBlock, args.model_dim, 2)
        self.inplanes = self.backbone.backbone_channels*8
        self.ls_grid_layer = self._make_resnet_layer(BasicBlock, args.model_dim, 1)
        self.inplanes = self.backbone.backbone_channels*2
        self.ms_grid_layer = self._make_resnet_layer(BasicBlock, args.model_dim//2, 1)
        self.inplanes = self.backbone.backbone_channels
        self.hs_grid_layer = self._make_resnet_layer(BasicBlock, args.model_dim//4, 1)
        self.fuse_grid_layer = self._make_resnet_layer(BasicBlock, args.model_dim, 1)
        # self.cam_layer = nn.Linear(431,1)

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )  # ,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        layers.append(nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, images, meta_lf_masks=None, meta_mf_masks=None, meta_hf_masks=None, is_train=False):
        x1, x2, x4 = self.backbone(images)

        batch, c, h, w = x1.shape
        device = x2.device

        # initialize token
        joint_token = self.joint_token_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        vertex_token = self.vertex_token_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        cam_token = self.cam_token_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        # process low scale grid feature
        ls_grid_feature = self.ls_grid_layer(x4).flatten(2)
        ls_grid_feature += self.ls_pos_embedding(batch, h // 8, w // 8, device).flatten(2)
        # process middle scale grid feature
        ms_grid_feature = self.ms_grid_layer(x2)
        ms_grid_feature += self.ms_pos_embedding(batch, h // 2, w // 2, device)
        ms_grid_feature = self.ms_patch_embed(ms_grid_feature)
        # process high scale grid feature
        hs_grid_feature = self.hs_grid_layer(x1)
        hs_grid_feature += self.hs_pos_embedding(batch, h , w , device)
        hs_grid_feature = self.hs_patch_embed(hs_grid_feature)

        fjv_token = torch.cat([ls_grid_feature.transpose(1, 2).contiguous(), joint_token, vertex_token], 1)
        if is_train==True:
            constant_tensor = torch.ones_like(fjv_token).cuda(device) * 0.01
            fjv_token = fjv_token * meta_lf_masks + constant_tensor * (1 - meta_lf_masks)
        # first stage
        input_feature = torch.cat(
            [cam_token, fjv_token], 1)  # BX456XC
        input_feature = input_feature + self.ls_fuse_pos_embed
        input_feature = self.ls_pos_drop(input_feature)
        # ls_output = ls_cam_parameter, ls_pred_3d_joints, ls_pred_3d_vertices_coarse, ls_pred_3d_vertices_mid, ls_pred_3d_vertices_fine
        ls_output = self.vim_block1(input_feature, 49)  # BX456XC+P
        ls_cjv_feature = torch.cat([ls_output[0].unsqueeze(1), ls_output[1], ls_output[2]], 1)
        ls_cjv_feature = self.fc1(ls_cjv_feature)
        # second stage
        ls_cam_features, ls_jv_features = ls_cjv_feature.split([1, ls_cjv_feature.shape[1] - 1], dim=1)
        ms_input_feature = torch.cat([ms_grid_feature, ls_jv_features], 1)  # BX456XC
        if is_train==True:
            constant_tensor = torch.ones_like(ms_input_feature).cuda(device) * 0.01
            ms_input_feature = ms_input_feature * meta_mf_masks + constant_tensor * (1 - meta_mf_masks)
        ms_input_feature = torch.cat(
            [ls_cam_features, ms_input_feature], 1)  # BX456XC
        ms_input_feature = ms_input_feature + self.ms_fuse_pos_embed
        ms_input_feature = self.ms_pos_drop(ms_input_feature)
        # ms_output = ms_cam_parameter, ms_pred_3d_joints, ms_pred_3d_vertices_coarse, ms_pred_3d_vertices_mid, ms_pred_3d_vertices_fine
        # ms_output = self.vim_block2(ms_input_feature, 196)  # BX456XC+P
        ms_cam_parameter, ms_pred_3d_joints, ms_pred_3d_vertices_coarse, ms_pred_3d_vertices_mid, ms_pred_3d_vertices_fine = self.vim_block2(ms_input_feature, 196)  # BX456XC+P
        # pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine = self.hgcn_block(
        #     ms_pred_3d_vertices_coarse)
        # ms_output = ms_cam_parameter, ms_pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine
        ms_output = ms_cam_parameter, ms_pred_3d_joints, ms_pred_3d_vertices_coarse, ms_pred_3d_vertices_mid, ms_pred_3d_vertices_fine

        ms_cjv_feature = torch.cat([ms_output[0].unsqueeze(1), ms_output[1], ms_output[2]], 1)
        ms_cjv_feature = self.fc2(ms_cjv_feature)
        # third stage
        ms_cam_features, ms_jv_features = ms_cjv_feature.split([1, ms_cjv_feature.shape[1] - 1], dim=1)
        hs_input_feature = torch.cat([hs_grid_feature, ms_jv_features], 1)  # BX456XC
        if is_train == True:
            constant_tensor = torch.ones_like(hs_input_feature).cuda(device) * 0.01
            hs_input_feature = hs_input_feature * meta_hf_masks + constant_tensor * (1 - meta_hf_masks)
        hs_input_feature = torch.cat(
            [ms_cam_features, hs_input_feature], 1)  # BX456XC
        hs_input_feature = hs_input_feature + self.hs_fuse_pos_embed
        hs_input_feature = self.hs_pos_drop(hs_input_feature)
        # hs_output = hs_cam_parameter, hs_pred_3d_joints, hs_pred_3d_vertices_coarse, hs_pred_3d_vertices_mid, hs_pred_3d_vertices_fine
        # hs_output = self.vim_block3(hs_input_feature, 196)  # BX456XC+P
        hs_cam_parameter, hs_pred_3d_joints, hs_pred_3d_vertices_coarse, hs_pred_3d_vertices_mid, hs_pred_3d_vertices_fine = self.vim_block3(hs_input_feature, 196)  # BX456XC+P
        # pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine = self.gcn_block(hs_pred_3d_vertices_coarse)
        # hs_output = hs_cam_parameter, hs_pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine
        hs_output = hs_cam_parameter, hs_pred_3d_joints, hs_pred_3d_vertices_coarse, hs_pred_3d_vertices_mid, hs_pred_3d_vertices_fine
        return ls_output, ms_output, hs_output

