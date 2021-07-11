import soundfile
import sys
#sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from IPython.display import display, Audio
from espnet2.bin.enh_inference import SeparateSpeech

sample_wav_sc, sr = soundfile.read("resources/record.wav")
#sample_wav_sc = sample_wav_mc[:,4]

enh_model_sc = SeparateSpeech(
  enh_train_config="conf/espnet2/enh/train_enh_rnn_tf.yaml",
  enh_model_file="resources/90epoch.pth",
  # for segment-wise process on long speech
  normalize_segment_scale=False,
)

wave = enh_model_sc(sample_wav_sc[None, ...], sr)
print("Input real noisy speech", flush=True)

audio = Audio(wave[0].squeeze(), rate=sr)
with open('enh_demo/test.wav', 'wb') as f:
  f.write(audio.data)
