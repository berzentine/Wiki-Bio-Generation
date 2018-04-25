#!/bin/sh

CUDA_AVAILABLE_DEVICES=0 python generation_dual_new_model_alignments_timestep.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset \
 --model_save_path ./saved_models/best_model_timestep.pth \
 --limit 0.05 \
 --cuda \
 --ref_path debug_ref_al.txt \
 --gen_path debug_gen_al.txt \
 --unk_gen_path debug_unk_al.txt
