"""
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
"""

import torch
import numpy as np
from torch import nn
from .module.transformer import *
from .module.position_encoding import build_position_encoding
from timm.models import create_model
from src.modeling.vim.models_mamba import feat_Mamba
from src.modeling._hgcn import HGCN, HyperGraphResBlock, HGGCN, HGGCNBlock
# from src.modeling._hgcn import HGCN, HyperGraphResBlock, HGGCNResBlock

from src.modeling._gcnn import GraphResBlock


class Vimfeat(nn.Module):
    def __init__(self, args, mesh_sampler, model_dim=(384,384), model_name="L"):
        super(Vimfeat, self).__init__()

        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = 14
        self.num_vertices = 431

        self.vim =  feat_Mamba(embed_dim=model_dim[1], depth=4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)

        # estimators
        self.xyz_regressor = nn.Linear(model_dim[1], 3)
        self.cam_predictor = nn.Linear(model_dim[1], 3)

    def forward(self, cjv_token, feat_num):
        device = cjv_token.device
        batch_size = cjv_token.size(0)


        cjv_features = self.vim(cjv_token).transpose(0, 1)

        cam_features, _, jv_features_2 = cjv_features.split([1, feat_num, cjv_features.shape[0] - 1 - feat_num], dim=0)

        # estimators
        cam_features = cam_features.transpose(0, 1).contiguous()
        pred_3d_coordinates = jv_features_2.transpose(0, 1).contiguous()
        pred_3d_joints = pred_3d_coordinates[:, :self.num_joints, :]
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.num_joints:, :]

        # coarse-to-fine mesh upsampling  # 431 -> 6890
        # pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        # pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        # return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine
        return cam_features, pred_3d_joints, pred_3d_vertices_coarse

class VimMeshRegressor(nn.Module):
    def __init__(self, args, mesh_sampler, model_dim=(384,384), model_name="L"):
        super(VimMeshRegressor, self).__init__()

        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = 14
        self.num_vertices = 431

        self.vim =  feat_Mamba(embed_dim=model_dim[1], depth=4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True)

        # estimators
        self.xyz_regressor = nn.Linear(model_dim[1], 3)
        self.cam_predictor = nn.Linear(model_dim[1], 3)

    def forward(self, cjv_token, feat_num):
        device = cjv_token.device
        batch_size = cjv_token.size(0)


        cjv_features = self.vim(cjv_token).transpose(0, 1)

        cam_features, _, jv_features_2 = cjv_features.split([1, feat_num, cjv_features.shape[0] - 1 - feat_num], dim=0)

        # estimators
        cam_parameter = self.cam_predictor(cam_features).view(batch_size, 3)
        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(0, 1).contiguous())
        pred_3d_joints = pred_3d_coordinates[:, :self.num_joints, :]
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.num_joints:, :]

        # coarse-to-fine mesh upsampling  # 431 -> 6890
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine


class HgcnMeshRegressor(nn.Module):
    def __init__(self, args, mesh_sampler, model_dim=(64,64)):
        super(HgcnMeshRegressor, self).__init__()

        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = 14
        self.num_vertices = 431


        # self.hgcn = HGCN(in_channels = 3, out_channels = model_dim[1], hidden_channels= model_dim[1])

        # self.mesh_type = 'body'
        # self.graph_conv = GraphResBlock(3, 3, model_dim[1], mesh_type=self.mesh_type)
        # self.hyper_graph_block = HyperGraphResBlock(3, 3, model_dim[1])
        # self.hggcn = HGGCN(3, model_dim[1], model_dim[1])
        self.hggcn_block = HGGCNBlock(3, 3, model_dim[1])
        # self.hggn_block = HGGCNResBlock(3, 3, model_dim[1])

        # estimators
        self.xyz_regressor = nn.Linear(model_dim[1], 3)

        self.incident_matrix = torch.load("tensor_hg_noverlap.pt").to(args.device)
        self.adjacency_matrix = torch.load("tensor_14joint_graph.pt").to(args.device)

    def forward(self, vertices_coord):
        device = vertices_coord.device
        batch_size = vertices_coord.size(0)


        # v_features_2 = self.hgcn(vertices_coord, self.incident_matrix, joint_coord)

        # pred_3d_vertices_coarse = self.hyper_graph_block(vertices_coord, self.incident_matrix)
        # pred_3d_vertices_feature = self.hggcn1(vertices_coord, self.incident_matrix, self.adjacency_matrix)
        pred_3d_vertices_coarse = self.hggcn_block(vertices_coord, self.incident_matrix, self.adjacency_matrix)
        # pred_3d_vertices_coarse = self.hggb_block(vertices_coord, self.incident_matrix, self.adjacency_matrix)

        # pred_3d_vertices_coarse = self.graph_conv(vertices_coord)
        # pred_3d_vertices_coarse = pred_3d_vertices_coarse + vertices_coord


        # estimators

        # pred_3d_coordinates = self.xyz_regressor(jv_features_2.contiguous())
        # pred_3d_coordinates = self.xyz_regressor(jv_features_2.contiguous())
        # pred_3d_vertices_coarse = pred_3d_coordinates[:, :self.num_vertices, :]
        # pred_3d_joints = pred_3d_coordinates[:, self.num_vertices:, :]

        # coarse-to-fine mesh upsampling  # 431 -> 6890
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine

class GcnMeshRegressor(nn.Module):
    def __init__(self, args, mesh_sampler, model_dim=(64, 64)):
        super(GcnMeshRegressor, self).__init__()
        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = 14
        self.num_vertices = 431
        self.mesh_type = 'body'
        self.graph_conv = GraphResBlock(3, 3, model_dim[1], mesh_type=self.mesh_type)
        # estimators
        self.xyz_regressor = nn.Linear(model_dim[1], 3)

        self.incident_matrix = torch.load("tensor_hg_noverlap.pt").to(args.device)
        self.adjacency_matrix = torch.load("tensor_14joint_graph.pt").to(args.device)

    def forward(self, vertices_coord):
        device = vertices_coord.device
        batch_size = vertices_coord.size(0)
        pred_3d_vertices_coarse = self.graph_conv(vertices_coord)
        # pred_3d_coordinates = self.xyz_regressor(jv_features_2.contiguous())
        # pred_3d_coordinates = self.xyz_regressor(jv_features_2.contiguous())
        # pred_3d_vertices_coarse = pred_3d_coordinates[:, :self.num_vertices, :]
        # pred_3d_joints = pred_3d_coordinates[:, self.num_vertices:, :]

        # coarse-to-fine mesh upsampling  # 431 -> 6890
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine