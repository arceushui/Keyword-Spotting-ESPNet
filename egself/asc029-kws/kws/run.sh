#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=6   # stage1-4: prepared data, stage:5 collect stats, stage:6 train kws, stage:7 decoding, stage:8 scoring
stop_stage=6
ngpu=1
nj=32

# Dataset
train_set=train_nodev_noise
valid_set=dev_noise
test_sets="eval-ch2"

# Model configure file
kws_config=conf/espnet2/kws/train_kws_transformer_v2.yaml

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=wav     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
frame_sample=16k      # Sampling rate.

# output dir
dumpdir=dump           # Directory to dump features.
expdir=exp-transformer-noise            # Directory to save experiments

export CUDA_VISIBLE_DEVICES=1

./kws.sh \
    --nj ${nj} \
    --inference_nj 32 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu} \
    --feats_type ${feats_type} \
    --audio_format ${audio_format} \
    --fs ${frame_sample} \
    --max_wav_duration 30 \
    --kws_config "${kws_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --dumpdir ${dumpdir} \
    --expdir ${expdir} \
    "$@"
