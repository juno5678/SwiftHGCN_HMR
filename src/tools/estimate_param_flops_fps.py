import torch
import time
import argparse
import sys
sys.path.insert(0,'/home/juno/MambaHMR')
from src.modeling.model.network import MambaHMR
from src.modeling._smpl import SMPL, Mesh
from thop import profile
from src.modeling.backbone.config import config as hrnet_config
from src.modeling.backbone.config import update_config as hrnet_update_config
from src.modeling.backbone.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.backbone.hrnet_32 import HigherResolutionNet
import torchvision.models as models
def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int,
                        help="adjust image resolution.")
    parser.add_argument("--image_file_or_path", default='./samples/human-body', type=str,
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--image_output_dir", default='demo/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
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
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--heatmap_loss_weight", default=100.0, type=float)

    parser.add_argument("--vloss_w_full", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub", default=0.33, type=float)
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w32',
                        help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--transformer_nhead", default=4, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                             "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--model_dim", default=384, type=int,
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
def run(model, img, name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():


        input = torch.unsqueeze(img, 0).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(100):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts - start_ts  # t_cnt + (end_raphormer_ts-start_ts)

    flops, params = profile(model, inputs=(input,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    print("=======================================")
    print("Model Name: " + name)
    print("FPS: %f" % (100 / t_cnt))
    print("=======================================")
    # 计算参数量
    num_params = 0
    for param in model.parameters():
        num_params += torch.prod(torch.tensor(param.size()))
        # print('模型参数量 1：', param.size())
    print('模型總参数量 1：', num_params)
    # 计算参数量
    num_params = 0
    for name, param in model.named_parameters():
        num_params += torch.prod(torch.tensor(param.size()))
        # print(name, param.size())
    print('模型参数量：', num_params)



if __name__ == "__main__":
    img = torch.randn(3, 224, 224)
    args = parse_args()
    mesh_sampler = Mesh()
    if args.arch == 'hrnet-w40':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
    elif args.arch == 'hrnet-w64':
        hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
    elif args.arch == 'hrnet-w32':
        backbone = HigherResolutionNet(args)
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    model = MambaHMR(args, mesh_sampler, backbone)
    run(model, img, 'MambaHMR')
    # run(backbone, img, 'MambaHMR')
