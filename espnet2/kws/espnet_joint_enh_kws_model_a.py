from distutils.version import LooseVersion
from functools import reduce
from itertools import permutations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import logging
import pickle
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)

from espnet2.kws.encoder.abs_encoder import AbsEncoder
from espnet2.kws.frontend.abs_frontend import AbsFrontend
from espnet2.kws.specaug.abs_specaug import AbsSpecAug
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
import torch.nn.functional as F

"""
"""

is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")

EPS = torch.finfo(torch.get_default_dtype()).eps

class ESPnetEnhKWSModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        enh: Optional[AbsEnhancement],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        enh_weight: float = 0.2,
        length_normalized_loss: bool = False,
        num_spk: int = 1,
    ):
        assert check_argument_types()
        assert 0.0 <= enh_weight <= 1.0, enh_weight

        super().__init__()
        self.vocab_size = vocab_size
        self.enh_weight = enh_weight
        self.token_list = token_list.copy()

        self.enh_model = enh
        self.num_spk = num_spk
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)

        self.mask_type = self.mask_type.upper() if self.mask_type else None
        assert self.loss_type in (
            # mse_loss(predicted_mask, target_label)
            "mask_mse",
            # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
            "magnitude",
            # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
            "spectrum",
            # si_snr(enhanced_waveform, target_waveform)
            "si_snr",
        ), self.loss_type

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder

        # mao
        self.linear = torch.nn.Linear(self.encoder.output_size() * 2, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, 
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        text_ref1: torch.Tensor,
        text_ref1_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Enhancement + Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_ref1_lengths.dim() == 1
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
            == text_ref1.shape[0]
            == text_ref1_lengths.shape[0]
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
            text_ref1.shape,
            text_ref1_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        text_length_max = text_ref1_lengths.max()
        text_ref1 = text_ref1[:, : text_length_max]
       
        # Enhancement moudle
        #0: Loss and mask predicted
        loss_enh, spectrum_pre, others = self.forward_enh(
             speech_mix, speech_mix_lengths, speech_ref1=speech_ref1,
        )

        if 0.0 < self.enh_weight < 1.0:

            #1:get predicted spectrums
            #predicted_spectrums = spectrum_pre
            feature_mix_complex, flens = self.enh_model.encoder(speech_mix, speech_mix_lengths)
            predicted_spectrums = feature_mix_complex * others['spk1']
            #2: generate enhanced wavs
            if predicted_spectrums is None:
                predicted_wavs = None
                raise ValueError("Predicted Wav is None !")
            elif isinstance(predicted_spectrums, list):
                # multi-speaker input
                predicted_wavs = [
                    self.enh_model.stft.inverse(ps, speech_mix_lengths)[0] for ps in predicted_spectrums
                ]
            else:
                # single-speaker input
                predicted_wavs = self.enh_model.stft.inverse(predicted_spectrums, speech_mix_lengths)[0]
       
            # KWS moudle
            feats, feats_lengths = predicted_wavs, speech_mix_lengths

            #1: Here, enhanced wavform pass asr extract feature and kws encoder
            encoder_out, encoder_out_lens = self.encode(feats, feats_lengths)
            text_ref_all = text_ref1
            text_ref_lengths = text_ref1_lengths
       
            #2: Compute kws loss
            mean = torch.mean(encoder_out, dim=1).unsqueeze(1)
            std = torch.std(encoder_out, dim=1).unsqueeze(1)
            hs_pad = torch.cat((mean, std), dim=-1) # (B, 1, D)
            ys_pred = self.linear(hs_pad)
            encoder_out_lens[:] = 2
            ys_pred = torch.cat([ys_pred[i, :l] for i, l in enumerate(encoder_out_lens)])
            ys_target =  torch.cat([text_ref_all[i, :l] for i, l in enumerate(text_ref_lengths)])
            loss_kws = self.criterion(ys_pred,ys_target)

            #3: Compute accuracy
            pred = ys_pred.data.max(1, keepdim=True)[1]
            correct = pred.eq(ys_target.data.view_as(pred)).sum()
            total = ys_target.size(0)
            self.acc = correct / total
       
        #4: Compute enh_kws model loss
        if self.enh_weight == 0.0:
            loss_enh = None
            loss = loss_kws

        elif self.enh_weight == 1.0:
            loss_kws = None
            loss = loss_enh
            self.acc = None
        else:
            loss = (1 - self.enh_weight) * loss_kws + self.enh_weight * loss_enh

        stats = dict(
            loss=loss.detach(),
            loss_kws=loss_kws.detach() if loss_kws is not None else None,
            loss_enh=loss_enh.detach() if loss_enh is not None else None,
            acc=self.acc,
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhancement + Frontend + Encoder 

        Args:
            speech_mix: (Batch, Length, ...)
            speech_mix_lengths: (Batch, )
        """
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
           
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # Enhancement moudle
        _, _, others = self.enh_model(speech_mix, speech_mix_lengths)
        #1:get predicted spectrums
        #predicted_spectrums = spectrum_pre
        feature_mix_complex, flens = self.enh_model.encoder(speech_mix, speech_mix_lengths)
        predicted_spectrums = feature_mix_complex * others['spk1']
        #2: generate enhanced wavs
        if predicted_spectrums is None:
            predicted_wavs = None
            raise ValueError("Predicted Wav is None !")
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.enh_model.stft.inverse(ps, speech_mix_lengths)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.enh_model.stft.inverse(predicted_spectrums, speech_mix_lengths)[0]

        # KWS moudle
        #0:. Encoder
        encoder_out, encoder_out_lens = self.encode(predicted_wavs, speech_mix_lengths)
       
        return encoder_out, encoder_out_lens

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}


    def _collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    )-> Dict[str, torch.Tensor]:

        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        # logging.info(f"**kwargs is {kwargs}")
        spectrum_pre, flens, others = self.enh_model(speech_mix, speech_mix_lengths) 
        predicted_spectrums = spectrum_pre

        #2: generate enhanced wavs
        if predicted_spectrums is None:
            predicted_wavs = None
            raise ValueError("Predicted Wav is None !")
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.enh_model.stft.inverse(ps, speech_mix_lengths)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.enh_model.stft.inverse(predicted_spectrums, speech_mix_lengths)[0]

        feats, feats_lengths = self._extract_feats(predicted_wavs, speech_mix_lengths)
        feats, feats_lengths = feats, feats_lengths 
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, freq)
            speech_lengths: (Batch, )
        """
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        #eps = 0 #1e-10
        #mean = feats.mean(0, keepdim=True)
        #std = feats.std(0, keepdim=True)
        #feats = (feats - mean) / (std + eps)
        feats = F.normalize(feats, p=2, dim=1)

        # 2. Data augmentation for spectrogram
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        #logging.info(f"in the espnet_joint_model3.py, kws module encoder input is feats is {feats}")
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    # Enhancement related, basicly from the espnet2/enh/espnet_model1.py
    def forward_enh(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        speech_ref = speech_ref[:, :, : speech_lengths.max()]
        speech_mix = speech_mix[:, : speech_lengths.max()]

        loss, spectrum_pre, others, out_lengths, perm = self.enh_compute_loss(
            speech_mix,
            speech_lengths,
            speech_ref,
            dereverb_speech_ref=None,
            noise_ref=None,
        )

        return loss, spectrum_pre, others


    def enh_compute_loss(
        self,
        speech_mix,
        speech_lengths,
        speech_ref,
        dereverb_speech_ref=None,
        noise_ref=None,
        cal_loss=True,
    ):

        """Compute loss according to self.loss_type.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            dereverb_speech_ref: (Batch, N, samples)
                        or (Batch, num_speaker, samples, channels)
            noise_ref: (Batch, num_noise_type, samples)
                        or (Batch, num_speaker, samples, channels)
            cal_loss: whether to calculate enh loss, defualt is True

        Returns:
            loss: (torch.Tensor) speech enhancement loss
            speech_pre: (List[torch.Tensor] or List[ComplexTensor])
                        enhanced speech or spectrum(s)
            others: (OrderedDict) estimated masks or None
            output_lengths: (Batch,)
            perm: () best permutation
        """

        #feature_mix, flens = self.enh_model.stft(speech_mix, speech_lengths)
        #feature_mix_complex = ComplexTensor(feature_mix[..., 0], feature_mix[..., 1]) 
        feature_mix_complex, flens = self.enh_model.encoder(speech_mix, speech_lengths)
        # enhancement model: encoder(stft) -> separator(rnn)
        feature_pre, flens, others = self.enh_model(speech_mix, speech_lengths)

        if self.loss_type != "si_snr":
            spectrum_mix = feature_mix_complex
            spectrum_pre = feature_pre

            if spectrum_pre is not None and not isinstance(
                spectrum_pre[0], ComplexTensor
            ):
                spectrum_pre = [
                    ComplexTensor(*torch.unbind(sp, dim=-1)) for sp in spectrum_pre
                ]

            if not cal_loss:
                loss, perm = None, None
                return loss, spectrum_pre, others, flens, perm


            # prepare reference speech and reference spectrum
            speech_ref = torch.unbind(speech_ref, dim=1)
            # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
            spectrum_ref = [self.enh_model.encoder(sr, speech_lengths)[0] for sr in speech_ref]

            # compute TF masking loss
            if self.loss_type.startswith("mask"):
                if self.loss_type == "mask_mse":
                    loss_func = self.tf_mse_loss
                else:
                    raise ValueError("Unsupported loss type: %s" % self.loss_type)

                assert others is not None
                mask_pre_ = [
                    others["spk{}".format(spk + 1)] for spk in range(self.num_spk)
                ]
                # prepare ideal masks
                mask_ref = self._create_mask_label(
                    spectrum_mix, spectrum_ref, mask_type=self.mask_type
                )

                # compute TF masking loss
                tf_loss, perm = self._permutation_loss(mask_ref, mask_pre_, loss_func)

            loss = tf_loss
            return loss, spectrum_pre, others, flens, perm

        else:
            raise ValueError("Unsupported loss type: %s" % self.loss_type)

    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        if not is_torch_1_3_plus:
            # in case of binary masks
            ref = ref.type(inf.dtype)
        diff = ref - inf
        if isinstance(diff, ComplexTensor):
            mseloss = diff.real ** 2 + diff.imag ** 2
        else:
            mseloss = diff ** 2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return mseloss

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm (torch.Tensor): specified permutation (batch, num_spk)
        Returns:
            loss (torch.Tensor): minimum loss with the best permutation (batch)
            perm (torch.Tensor): permutation for inf (batch, num_spk)
                                 e.g. tensor([[1, 0, 2], [0, 1, 2]])
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        if perm is None:
            device = ref[0].device
            all_permutations = list(permutations(range(num_spk)))
            losses = torch.stack([pair_loss(p) for p in all_permutations], dim=1)
            loss, perm = torch.min(losses, dim=1)
            perm = torch.index_select(
                torch.tensor(all_permutations, device=device, dtype=torch.long),
                0,
                perm,
            )
        else:
            loss = torch.tensor(
                [
                    torch.tensor(
                        [
                            criterion(
                                ref[s][batch].unsqueeze(0), inf[t][batch].unsqueeze(0)
                            )
                            for s, t in enumerate(p)
                        ]
                    ).mean()
                    for batch, p in enumerate(perm)
                ]
            )

        return loss.mean(), perm

    @staticmethod
    def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        Args:
            mix_spec: ComplexTensor(B, T, F)
            ref_spec: List[ComplexTensor(B, T, F), ...]
            mask_type: str
        Returns:
            labels: List[Tensor(B, T, F), ...] or List[ComplexTensor(B, T, F), ...]
        """

        # Must be upper case
        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "PSM^2",
        ], f"mask type {mask_type} not supported"
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + EPS)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + EPS)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + EPS)
                phase_mix = mix_spec / (abs(mix_spec) + EPS)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + EPS)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_type == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + EPS)
                phase_mix = mix_spec / (abs(mix_spec) + EPS)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + EPS)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label
 
