#!/bin/sh

CUDA_AVAILABLE_DEVICES=0 python generation_dual_new_model_alignments_timestep.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset-debug \
 --model_save_path ./saved_models/best_model_debug_timestep_changed.pth \
 --limit 0.05 \
 --cuda \
 --use_pickle \
 --alignments ./data/Wiki-Data/alignments/ibm_model1_alignments.txt \
 --ref_path debug_ref_al_t.txt \
 --gen_path debug_gen_al_t.txt \
 --unk_gen_path debug_unk_al_t.txt
