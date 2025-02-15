#!/bin/sh
python tools/main_track.py \
    --det_conf 0.6 \
    --act 0.8 \
    --inact 0.9 \
    --inact_patience 50 \
    --combi sum_0.6 \
    --do_inact 0 \
    --on_the_fly 1 \
    --store_feats 0 \
    --det_file yolox_dets.txt \
    --config_path config/config_tracker_dance.yaml \
    --splits dance_val \
    --new_track_conf 0.6 \
    --len_thresh 0 \
    --remove_unconfirmed 0 \
    --last_n_frames 5
