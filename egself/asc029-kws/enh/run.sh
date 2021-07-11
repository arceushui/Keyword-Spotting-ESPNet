#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

nj=120
inference_nj=10
stage=7
stop_stage=9

# Dataset
train_set=train_nodev
valid_set=dev
test_sets="eval"

# Model configure file
enh_config=conf/espnet2/enh/train_enh_rnn_tf.yaml

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=wav     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
sample_rate=16k      # Sampling rate.

# Output
dumpdir=dump
expdir=exp

./enh.sh \
    --nj ${nj} \
    --inference_nj ${inference_nj} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --fs ${sample_rate} \
    --feats_type ${feats_type} \
    --ngpu 1 \
    --spk_num 1 \
    --audio_format ${audio_format} \
    --enh_config ${enh_config} \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    --dumpdir ${dumpdir} \
    --expdir ${expdir} \
    "$@"
