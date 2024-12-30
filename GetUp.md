# Train运行命令
```shell
mpiexec -n 4 python image_train.py \
        --data_dir 'F:\Datasets\RainH_without_mask\Train' \
        --data_val_dir 'F:\Datasets\RainH_without_mask\Test' \
        --log_dir  'F:\Datasets\RainH_without_mask\log' \
        --batch_size 64 \
        --image_size 64 \
        --lr 1e-4 \
        --ema_rate 0.9999 \
        --log_interval 10 \
        --save_interval 10000 \
        --attention_resolutions 32,16,8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --use_fp16 True \
        --resblock_updown True \
        --use_scale_shift_norm True \
        --lr_anneal_steps 100000
```
## if memory out of range 
```shell
mpiexec -n 4 python scripts/image_train.py \
        --data_dir 'F:\Datasets\RainH_without_mask\Train' \
        --data_val_dir 'F:\Datasets\RainH_without_mask\Test' \
        --log_dir  'F:\Datasets\RainH_without_mask' \
        
        --batch_size 256 \
        --microbatch 1 \
        --image_size 256 \
        
        --lr 1e-4 \
        --anneal_lr True \
        --weight_decay 0.001 \
        --ema_rate 0.9999 \
        
        --log_interval 1000 \
        --save_interval 10000 \
        
        --attention_resolutions 8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --resblock_updown True \
        --use_fp16 True \
        --use_scale_shift_norm True \
        --lr_anneal_steps 500000
```

## other Hyperparameters settings
```shell
  --batch_size 256,128,64 
  --image_size 256,64,32 
  --attention_resolutions 32,16,8
  --weight_decay 0.001 
  --resblock_updown True,False
```
python image_train.py --data_dir 'F:\Datasets\RainH_without_mask\Train' --data_val_dir 'F:\Datasets\RainH_without_mask\Test' --log_dir  'F:\Datasets\RainH_without_mask\log' --batch_size 8 --image_size 128 --lr 1e-4 --ema_rate 0.9999 --log_interval 100 --save_interval 1000 --attention_resolutions 32,16,8 --num_channels 256 --num_heads 4 --num_res_blocks 2 --use_fp16 True --resblock_updown True --use_scale_shift_norm True --lr_anneal_steps 100000


# Sample运行命令
```shellx
python image_sample.py \
        --data_path 'F:\Datasets\RainH_without_mask\Train' \
        --model_path 'F:\Datasets\RainH_without_mask\log\xxxxx.pt' \
        --log_dir  'F:\Datasets\RainH_without_mask\log' \
        --batch_size 8 \
        --image_size 128 \
        --attention_resolutions 32,16,8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --use_fp16 True \
        --resblock_updown True \
        --use_scale_shift_norm True
```

```shellx
python image_sample.py \
        --data_path 'F:\Datasets\RainH_without_mask\Test\in' \
        --model_path 'F:\Datasets\RainH_without_mask\logs\1_4090\model040000.pt' \
        --log_dir  'F:\Datasets\RainH_without_mask\Test\log' \
        --batch_size 16 \
        --image_size 64 \
        --attention_resolutions 32,16,8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --use_fp16 True \
        --resblock_updown True \
        --use_scale_shift_norm True
```


```shellx
python image_train.py \
        --data_dir './Datasets/RainH_without_mask/Train' \
        --data_val_dir './Datasets/RainH_without_mask/Test' \
        --log_dir  './Datasets/RainH_without_mask/log' \
        --batch_size 8 \
        --image_size 256 \
        --patch_size (256,256,3) \
        --patch_overlap (8, 8, 0) \
        --lr 1e-4 \
        --ema_rate 0.9999 \
        --log_interval 100 \
        --save_interval 1000 \
        --attention_resolutions 32,16,8 \
        --num_channels 256 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --use_fp16 True \
        --resblock_updown True \
        --use_scale_shift_norm True \
        --lr_anneal_steps 100000
```
