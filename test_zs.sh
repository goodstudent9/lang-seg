for fold in 0 1 2 3; do
python -u test_lseg_zs.py --backbone clip_resnet101 --module clipseg_DPT_test_v2 --dataset coco \
--widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold ${fold} --nshot 0 \
--weights checkpoints/pascal_fold${fold}.ckpt 
done