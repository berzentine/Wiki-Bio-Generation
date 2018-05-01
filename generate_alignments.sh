#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python generation_dual_new_model_alignments.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset-debug \
 --limit 0.05 \
 --model_save_path ./saved_models/best_model_debug_changed.pth \
 --alignments ./data/Wiki-Data/alignments/ibm_model1_alignments.txt \
 --alignments_pickle ./data/Wiki-Data/alignments/ibm_model1_alignments.pickle \
 --ref_path full_ref_al.txt \
 --gen_path full_gen_al.txt \
 --unk_gen_path full_unk_al.txt \
 --use_pickle \
 --cuda
