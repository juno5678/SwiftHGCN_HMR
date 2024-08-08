#python -m torch.distributed.launch --nproc_per_node=1 \
python src/tools/run_phmr_bodymesh.py \
       --train_yaml /home/user/juno/datasets/3dpw/train.yaml \
       --val_yaml /home/user/juno/datasets/3dpw/test_has_gender.yaml \
       --arch "hrnet-w32" \
       --num_workers 10 \
       --per_gpu_train_batch_size 64 \
       --per_gpu_eval_batch_size 64 \
       --model_dim 384 \
       --position_dim 128 \
       --dropout 0.1 \
       --num_train_epochs 5 \
       --learning_rate 1e-3 \
       --resume_checkpoint /home/user/juno/MambaHMR/backup/mfvjm_mms_refine_hgcn/pt/checkpoint-23-171074/state_dict.bin
