#!/bin/sh
CUDA_AVAILABLE_DEVICES=0 python main_dual_alignments_timestep.py \
--epochs 500 \
--model_save_path dual_alignment \
--plot_save_path dual_alignment \
--data ./data/Wiki-Data/wikipedia-biography-dataset-dummy/ \
--use_cosine True \
--alignments ./data/Wiki-Data/alignments-debug/alignments.txt
