#!/bin/sh

CUDA_AVAILABLE_DEVICES=0 python generation_dual_new_model_alignments_timestep.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset-debug \
 --model_save_path saved_models/dual_alignment \
 --limit 0.005 \
 --ref_path debug_ref.txt \
 --gen_path debug_gen.txt \
 --unk_gen_path debug_unk.txt
