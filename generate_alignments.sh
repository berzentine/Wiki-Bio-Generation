#!/bin/sh

CUDA_AVAILABLE_DEVICES=0 python generation_dual_new_model_alignments.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset \
 --limit 0.05 \
 --model_save_path ./saved_models/best_model_timestep_changed.pth \
 --ref_path full_ref_al.txt \
 --gen_path full_gen_al.txt \
 --unk_gen_path full_unk_al.txt \
 --use_pickle \
 --cuda
