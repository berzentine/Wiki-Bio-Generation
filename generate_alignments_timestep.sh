#!/bin/sh
CUDA_AVAILABLE_DEVICES=1 python generation_dual_new_model_alignments_timestep.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset \
 --model_save_path ./saved_models/best_model_full_ibm1.pth \
 --limit 0.05 \
 --cuda \
 --use_pickle \
 --alignments ./data/Wiki-Data/alignments/ibm_model1_alignments.txt \
 --alignments_pickle ./data/Wiki-Data/alignments/ibm_model1_alignments.pickle \
 --ref_path full_ref_al_ibm1t.txt \
 --gen_path full_gen_al_ibm1t.txt \
 --unk_gen_path full_unk_al_ibm1t.txt
