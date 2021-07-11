#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.enh_kws_a import EnhKWSTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none

def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)

class Speech2Text:
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        joint_train_config: Union[Path, str],
        joint_model_file: Union[Path, str] = None,
        token_type: str = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
    ):
        assert check_argument_types()

        # 1. Build Joint model
        scorers = {}
        joint_model, joint_train_args = EnhKWSTask.build_model_from_file(
            joint_train_config, joint_model_file, device
        )

        #joint_model.eval()
        joint_model.to(dtype=getattr(torch, dtype)).eval()
 
        logging.info(f"Decoding device={device}, dtype={dtype}")

        token_list = joint_model.token_list

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        if token_type is None:
            token_type = joint_train_args.token_type

        if token_type is None:
            tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        #converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        self.joint_model = joint_model
        self.joint_train_args = joint_train_args
        #self.converter = converter
        self.tokenizer = tokenizer
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.linear = torch.nn.Linear(self.joint_model.encoder.output_size() * 2, 2)

    @torch.no_grad()
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray],
    ) -> List[List[Tuple[float, Optional[str], List[str], List[int], Hypothesis]]]:
        """Inference

        Args:
            speech_mix: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()
        speech = speech_mix

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lenghts: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)
        #logging.info(f"'speech',{batch['speech'].shape}, and speech_lengths is {batch['speech_lengths'].shape}")

        #enc, _ = self.joint_model.encode(batch['speech'],batch['speech_lengths']) 
        enc, _ = self.joint_model.inference(batch['speech'],batch['speech_lengths']) 
        #logging.info(f"in the Speech2Text __call__ function encode output  is {enc}")
        #results_list=[]        
        assert len(enc) == 1, len(enc)

        mean = torch.mean(enc, dim=1).unsqueeze(1)
        std = torch.std(enc, dim=1).unsqueeze(1)
        hs_pad = torch.cat((mean, std), dim=-1) # (B, 1, D)
        ys_pred = self.linear(hs_pad)

        lpz = ys_pred.squeeze(0) # shape of (T, D)
        idx = lpz.argmax(-1).cpu().numpy().tolist()
        hyp = {}
        hyp['yseq'] = idx
        return hyp

def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    joint_train_config: str,
    joint_model_file: str,
    token_type: Optional[str],
    allow_variable_data_keys: bool,
    normalize_output_wav: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        joint_train_config=joint_train_config,
        joint_model_file=joint_model_file,
        token_type=token_type,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
    )

    # 3. Build data-iterator
    loader = EnhKWSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=EnhKWSTask.build_preprocess_fn(speech2text.joint_train_args, False),
        collate_fn=EnhKWSTask.build_collate_fn(speech2text.joint_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]]


            # Only supporting batch_size==1
            key = keys[0]
            # Create a directory: outdir/{n}best_recog
            ibest_writer = writer[f"best_recog"]

            # Write the result to each file
            ibest_writer["token_int"][key] = " ".join(map(str, results['yseq']))

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="EnhKWS Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--joint_train_config", type=str, required=True)
    group.add_argument("--joint_model_file", type=str, required=True)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group = parser.add_argument_group("Output wav related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Weather to normalize the predicted wav to [-1~1]",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
