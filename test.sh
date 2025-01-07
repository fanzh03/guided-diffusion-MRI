#!/bin/bash


### training ###
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# python /ghome/fanzh/IRODE/RFODE_code/train.py -opt=/ghome/fanzh/IRODE/RFODE_code/rf-NAFNet.yml
python /ghome/fanzh/guided-diffusion-MRI/image_sample.py \
        --data_dir '/gdata2/fanzh/nii_data/T1_val/3T' \
        --model_path '/gdata2/fanzh/MRI/log40/model200000.pt' \
        --log_dir  '/gdata2/fanzh/MRI/log_result2' \
        --batch_size 4 \
        --image_size 256 \
        --attention_resolutions 32,16,8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 1 \
        --use_fp16 True \
        --resblock_updown True \
        --use_scale_shift_norm True \
        --data_shuffle False \
        # --lr 1e-4 \
        # --lr_anneal_steps 100000
        # --patch_size 256,256,3 \
        # --patch_overlap 8,8,0 \
        # --ema_rate 0.9999 \
        # --log_interval 100 \
        # --save_interval 5000 \
