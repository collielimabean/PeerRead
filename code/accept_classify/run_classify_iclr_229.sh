#!/bin/bash

DDIR=../../data
DATADIR=("$DDIR/iclr_2017")
DATASETS=("train" "dev" "test")
FEATDIR=dataset
MAX_VOCAB=False
ENCODER="w2v"
HAND=True

# # run baseline models
# echo "Classigying..." $DATADIR $DATASET $ENCODER $MAX_VOCAB $HAND
# python3 classify_full_result_229.py \
#   $DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
#   $DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
#   $DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
#   $DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
#   $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat \
# > ../accept_classify/result/result_iclr_full.txt
#exit

echo "Classigying..." $DATADIR $DATASET $ENCODER $MAX_VOCAB $HAND
python3 classify_229.py \
  $DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
  $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat \
> ../accept_classify/result/result_iclr.txt
#exit
