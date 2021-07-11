import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.kws.encoder.abs_encoder import AbsEncoder
from espnet2.kws.encoder.rnn_encoder import RNNEncoder
from espnet2.kws.encoder.transformer_encoder import TransformerEncoder
from espnet2.kws.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.kws.frontend.abs_frontend import AbsFrontend
from espnet2.kws.frontend.default import DefaultFrontend
from espnet2.kws.frontend.windowing import SlidingWindow

from espnet2.kws.specaug.abs_specaug import AbsSpecAug
from espnet2.kws.specaug.specaug import SpecAug
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.nets.beamformer_net import BeamformerNet
from espnet2.enh.nets.tasnet import TasNet

from espnet2.enh.nets.tf_mask_net_a import TFMaskingNet

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor_kws import CommonPreprocessor_multi
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from espnet2.kws.espnet_joint_enh_kws_model_a import ESPnetEnhKWSModel

from espnet2.kws.encoder.conformer_encoder import ConformerEncoder
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task_enh_kws import AbsTask

enh_choices = ClassChoices(
    name="enh",
    classes=dict(tf_masking_a=TFMaskingNet, tasnet=TasNet, wpe_beamformer=BeamformerNet),
    type_check=AbsEnhancement,
    default="tf_masking_a",
)

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend, sliding_window=SlidingWindow),
    type_check=AbsFrontend,
    default="default",
)

specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)

normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)

MAX_REFERENCE_NUM = 100

"""
2021-2-22
here ,the model is joint enh and kws module, and pre-trained enhanced model and plus kws module, 
1. load pre-trained enh model and don't update parater of enh module , only updated parameter of KWS module,
"""
class EnhKWSTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --enh and --enh_conf
        enh_choices,
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --encoder and --encoder_conf
        encoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--enh_kws_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhKWSModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="word",
            choices=["char", "word"],
            help="The text will be tokenized " "in the specified level token",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        # TODO(Jing): ask Kamo if it ok to support several args,
        # like text_name = 'text_ref1' and 'text_ref2'
        if args.use_preprocessor:
            retval = CommonPreprocessor_multi(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                text_name=["text_ref1"],
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "speech_ref1", "text_ref1")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref"]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["text_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhKWSModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 0. Build pre enhancement model
        enh_model = enh_choices.get_class(args.enh)(**args.enh_conf)

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Build model
        model = ESPnetEnhKWSModel(
            vocab_size=vocab_size,
            enh=enh_model,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            token_list=token_list,
            **args.enh_kws_model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
