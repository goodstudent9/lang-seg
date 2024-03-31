#!/bin/bash
#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384
export PYTHONPATH=$PYTHONPATH:/home/lang-seg/modules/models

python -m pdb train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 2 --exp_name lseg_ade20k_l16 \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

# python -m pdb train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 16 --exp_name lseg_ade20k_l16 \
# --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384