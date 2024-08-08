#python -m torch.distributed.launch --nproc_per_node=1 \
#       --use_env\
#/home/user/juno/datasets/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
#       --train_yaml /home/juno/datasets/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
#--train_yaml /home/juno/datasets/up3d/trainval.yaml \
#--val_yaml /home/juno/datasets/3dpw/test_has_gender.yaml \
#       --val_yaml /home/juno/datasets/human3.6m/valid.protocol2.yaml \
python src/tools/run_phmr_bodymesh.py \
       --train_yaml /home/juno/datasets//Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml\
       --val_yaml /home/juno/datasets/human3.6m/valid.protocol2.yaml \
       --arch "hrnet-w48" \
       --num_workers 10 \
       --per_gpu_train_batch_size 24 \
       --per_gpu_eval_batch_size 24 \
       --model_dim 384 \
       --position_dim 128 \
       --dropout 0.1 \
       --learning_rate 1e-3 \
       --num_train_epochs 40 \
       --drop-path 0.05 \
       --vertices_loss_weight 200 \
       --joints_loss_weight 1000 \
       --edge_loss_weight 50 \
       --normal_loss_weight 50 \
       --resume_checkpoint /home/juno/MambaHMR/models/mfvjm_mms_refine_hgcn/h36m/checkpoint-23-171074/state_dict.bin
#       --refine_vertices_loss_weight 200 \
#       --refine_joint_loss_weight 1000

#       --resume_checkpoint /home/user/juno/MambaHMR/backup/mulit_scale/checkpoint-0-0/state_dict.bin


