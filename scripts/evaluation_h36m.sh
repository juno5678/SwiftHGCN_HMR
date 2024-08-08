#python -m torch.distributed.launch --nproc_per_node=1 \
#	  --use_env \
python    src/tools/run_phmr_bodymesh.py \
          --val_yaml /home/juno/datasets/human3.6m/valid.protocol2.yaml \
          --arch "hrnet-w32" \
          --num_workers 4 \
          --per_gpu_eval_batch_size 12 \
          --model_dim 384 \
          --position_dim 128 \
          --dropout 0.1 \
          --run_eval_only \
          --resume_checkpoint /home/juno/MambaHMR/models/mfvjm_mms_refine_hgcn/h36m/checkpoint-23-171074/state_dict.bin


