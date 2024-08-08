python -m torch.distributed.launch --nproc_per_node=1 \
       --use_env\
       src/tools/run_phmr_bodymesh.py \
       --train_yaml /home/juno/datasets/Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml /home/juno/datasets/human3.6m/valid.protocol2.yaml \
       --num_workers 8 \
       --per_gpu_train_batch_size 24 \
       --per_gpu_eval_batch_size 24 \
       --model_dim 512 \
       --position_dim 128 \
       --dropout 0.1 \
       --learning_rate 1e-5 \
       --num_train_epochs 60

