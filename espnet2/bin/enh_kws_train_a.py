#!/usr/bin/env python3

# enh_kws different from enh_kws_a: enh_kws can process 8k telephone noise, enh_kws_a can enhance 16k other type noise.

from espnet2.tasks.enh_kws_a import EnhKWSTask

def get_parser():
    parser = EnhKWSTask.get_parser()
    return parser

def main(cmd=None):
    r"""ENH-KWS training.

    Example:

        % python enh_kws_train.py asr --print_config --optim adadelta \
                > conf/train_enh_kws.yaml
        % python enh_kws_train.py --config conf/train_enh_kws.yaml
    """
    EnhKWSTask.main(cmd=cmd)

if __name__ == "__main__":
    main()
