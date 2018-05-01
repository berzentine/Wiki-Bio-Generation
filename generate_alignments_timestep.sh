#!/bin/sh
CUDA_AVAILABLE_DEVICES=1 python generation_dual_new_model_alignments_timestep.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset-debug \
 --model_save_path ./saved_models/best_model_debug_edit.pth \
 --limit 0.05 \
 --cuda \
 --use_pickle \
 --alignments ./data/Wiki-Data/alignments/combined_alignments_p0_e1.txt \
 --alignments_pickle ./data/Wiki-Data/alignments/combined_alignments_p0_e1.pickle \
 --ref_path debug_ref_al_et.txt \
 --gen_path debug_gen_al_et.txt \
 --unk_gen_path debug_unk_al_et.txt
