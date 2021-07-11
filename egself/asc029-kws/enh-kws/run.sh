#!/bin/bash
echo
echo "$0 $@"
echo
echo `date`
echo

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=2   # stage2: speed perturb, stage3: format wav.scp, stage6:collect stats, stage7:training, stage8:decoding, stage9:scoring
stop_stage=9
ngpu=1
nj=100

joint_config=conf/espnet2/enh-kws/train_enh_kws_transformer.yaml

feats_type=raw
audio_format=wav
fs=16k

train_set=train_nodev
valid_set=dev
test_sets="eval eval-ch2"

dumpdir=dump               # Directory to dump features.
expdir=exp-transformer     # Directory to save experiments.

. utils/parse_options.sh

export CUDA_VISIBLE_DEVICES=1
 
./enh_kws.sh    \
    --stage $stage \
    --stop_stage $stop_stage \
    --feats_type $feats_type \
    --fs $fs \
    --audio_format $audio_format \
    --dumpdir "$dumpdir" \
    --expdir  "$expdir" \
    --ngpu $ngpu \
    --max_wav_duration 30 \
    --joint_config  "${joint_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" 

