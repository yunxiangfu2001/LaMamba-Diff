#!/bash/bin
python sample.py --model LaMamba-Diff --image-size 512 \
--ckpt ckpt_path \
--cfg configs/LaMamba-DIff-XL.yaml \
--num-classes 1000 --num-heads 16 --image-size 512 --cfg-scale 4.0
