# Keyword Spotting Alibaba

This project is built using [Espnet2](https://github.com/espnet/espnet).

## Installation
- If you intend to do full experiments including DNN training, then see [Installation](https://espnet.github.io/espnet/installation.html). In our project, we compiled our own kaldi although it is not required by espnet2 (Use this method to install the espnet2 in out project)

- If you just need the Python module only:
    ```sh
    pip install espnet
    # To install latest
    # pip install git+https://github.com/espnet/espnet
    ```

    You need to install some packages.

    ```sh
    pip install torch
    pip install chainer==6.0.0 cupy==6.0.0    # [Option] If you'll use ESPnet1
    pip install torchaudio                    # [Option] If you'll use enhancement task
    pip install torch_optimizer               # [Option] If you'll use additional optimizers in ESPnet2
    ```

    There are some required packages depending on each task other than above. If you meet ImportError, please intall them at that time.
- Once installed, run `wandb login` to enable tracking runs using W&B. 

## Dataset

### Wukong wukong dataset

- See the [Dataset](https://entuedu-my.sharepoint.com/:f:/g/personal/pang0208_e_ntu_edu_sg/Ep22QUyPlV9IuZsFJB4_Cq0BLKoS6DpM1M4_i5pKnRxsNg?e=6vT0sH)
- Extract all the tar files in the shared folder.
- README.pdf has some details about the dataset.

## Usage

### Espnet2 Tutorial (It is just a reference to the toolkit which is not required by our project)
See [Usage](https://espnet.github.io/espnet/tutorial.html).

### Keyword Spotting Transformer
1. Change to the directory below.

```sh
cd egself/asc029-kws/kws/    
```

2. Read `README.md` for more usage.

### Speech Enhancement Bi-LSTM
1. Change to the directory below.

```sh
cd egself/asc029-kws/enh/    
```

2. Read `README.md` for more usage.

### Joint-training Speech Enhancement & Keyword Spotting
1. Change to the directory below.

```sh
cd egself/asc029-kws/enh-kws/
```

2. Read `README.md` for more usage.

## References

[1] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-End Speech Processing Toolkit," *Proc. Interspeech'18*, pp. 2207-2211 (2018)

[2] Suyoun Kim, Takaaki Hori, and Shinji Watanabe, "Joint CTC-attention based end-to-end speech recognition using multi-task learning," *Proc. ICASSP'17*, pp. 4835--4839 (2017)

[3] Shinji Watanabe, Takaaki Hori, Suyoun Kim, John R. Hershey and Tomoki Hayashi, "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, Dec. 2017

## Citations

```
@inproceedings{watanabe2018espnet,
  author={Shinji Watanabe and Takaaki Hori and Shigeki Karita and Tomoki Hayashi and Jiro Nishitoba and Yuya Unno and Nelson {Enrique Yalta Soplin} and Jahn Heymann and Matthew Wiesner and Nanxin Chen and Adithya Renduchintala and Tsubasa Ochiai},
  title={{ESPnet}: End-to-End Speech Processing Toolkit},
  year={2018},
  booktitle={Proceedings of Interspeech},
  pages={2207--2211},
  doi={10.21437/Interspeech.2018-1456},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1456}
}
@inproceedings{hayashi2020espnet,
  title={{Espnet-TTS}: Unified, reproducible, and integratable open source end-to-end text-to-speech toolkit},
  author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},
  booktitle={Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7654--7658},
  year={2020},
  organization={IEEE}
}
@inproceedings{inaguma-etal-2020-espnet,
    title = "{ESP}net-{ST}: All-in-One Speech Translation Toolkit",
    author = "Inaguma, Hirofumi  and
      Kiyono, Shun  and
      Duh, Kevin  and
      Karita, Shigeki  and
      Yalta, Nelson  and
      Hayashi, Tomoki  and
      Watanabe, Shinji",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.34",
    pages = "302--311",
}
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
```
