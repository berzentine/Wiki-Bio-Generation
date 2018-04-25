#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python generation_dual_new_model.py \
 --data ./data/Wiki-Data/wikipedia-biography-dataset-debug \
 --limit 0.05 \
 --ref_path debug_ref.txt \
 --gen_path debug_gen.txt \
 --unk_gen_path debug_unk.txt \
 --use_pickle \
 --cuda
