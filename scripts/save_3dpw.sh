
#python -m torch.distributed.launch --nproc_per_node=2 \
python    src/tools/run_phmr_bodymesh_inference_dataset.py \
          --val_yaml /home/juno/datasets/3dpw/test_has_gender.yaml \
          --arch "hrnet-w32" \
          --num_workers 12 \
          --per_gpu_train_batch_size 24 \
          --per_gpu_eval_batch_size 24 \
          --model_dim 384 \
          --position_dim 128 \
          --dropout 0.1 \
          --num_train_epochs 10 \
          --run_eval_only \
          --resume_checkpoint /home/juno/MambaHMR/models/mfvjm_mms_refine/3DPW/checkpoint-4-1420/model.bin