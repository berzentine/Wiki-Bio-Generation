#!/bin/sh
#CUDA_AVAILABLE_DEVICES=0 python main_dual_alignments_timestep.py \
#--epochs 500 \
#--model_save_path dual_alignment \
#--plot_save_path dual_alignment \
#--data ./data/Wiki-Data/wikipedia-biography-dataset-dummy/ \
#--use_cosine True \
#--alignments ./data/Wiki-Data/alignments-debug/alignments.txt


CUDA_VISIBLE_DEVICES=1 \
python generation_dual_new_model_alignments_timestep.py  \
--cuda --data ./data/Wiki-Data/wikipedia-biography-dataset-debug/ \
--model_save_path ./saved_models/sota_structure_aware_debug.pth \
--plot_save_path ./saved_models/sota_structure_aware_debug.png \
--ref_path reference_sota_structure_aware_debug.txt \
--gen_path generated_sota_structure_aware_debug.txt \
--unk_gen_path unk_sota_structure_aware_debug.txt \
--alignments_pickle ./data/Wiki-Data/alignments/combined_alignments_p1_e1.pickle
--use_pickle
--use_alignments False
