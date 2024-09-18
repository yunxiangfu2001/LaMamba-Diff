#!/bash/bin
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29588 train.py --model LaMamba-Diff \
--data-path /data/dataset/imagenet/train \
--results-dir output/LaMamba-Diff-XL_imagenet512 \
--window-size 8 --num-heads 16 \
--ckpt-every 50000 --global-batch-size 256 --num-classes 1000 --epochs 500 \
--cfg configs/LaMamba-DIff-XL.yaml --image-size 512

