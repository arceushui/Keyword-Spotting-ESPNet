#!/usr/bin/env python3
from espnet2.tasks.kws import KWSTask


def get_parser():
    parser = KWSTask.get_parser()
    return parser


def main(cmd=None):
    r"""KWS training.

    Example:

        % python kws_train.py kws --print_config --optim adadelta \
                > conf/train_kws.yaml
        % python kws_train.py --config conf/train_kws.yaml
    """
    KWSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
