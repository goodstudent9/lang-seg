export CUDA_VISIBLE_DEVICES=1; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset coco --data-path ../datasets/ \
--weights /home/lang-seg/checkpoints/lseg_ade20k_l16/version_1/checkpoints/last.ckpt --test-batch-size 32 --widehead --no-scaleinv --not_changed True \
--no-strict


