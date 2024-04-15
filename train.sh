#!/bin/bash
#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384
export PYTHONPATH=$PYTHONPATH:/home/lang-seg/modules/models

python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 1 --exp_name lseg_ade20k_l16 --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --not_changed True

python train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 2 --exp_name lseg_ade20k_l16 --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --not_changed False
torchrun --nproc_per_node=gpu train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 2 --exp_name lseg_ade20k_l16 --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --not_changed False

(Pdb) b /home/lang-seg/modules/lsegmentation_module.py:39
Breakpoint 1 at /home/lang-seg/modules/lsegmentation_module.py:39
(Pdb) b /home/lang-seg/modules/lseg_module.py:84