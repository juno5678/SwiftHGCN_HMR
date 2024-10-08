python ./src/tools/run_phmr_bodymesh_inference.py \
       --resume_checkpoint /home/juno/MambaHMR/backup/pretrain_feat_vim/47-58/checkpoint-8-119008/state_dict.bin \
       --image_file_or_path ./samples \
       --image_output_dir ./demo \
       --model_dim 384 \
       --position_dim 128 \
       --dropout 0.1 \
       --learning_rate 1e-5 \
       --num_train_epochs 2 \
       --drop-path 0.05 \
