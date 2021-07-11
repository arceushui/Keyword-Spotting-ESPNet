#!/usr/bin/env python3

from espnet2.tasks.enh_kws import EnhKWSTask

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
