# Joint training speech enhancement & keyword spotting

## Usage

1. Add required path names in `path.sh`

```sh
# Absolute path of this project root. See example below.
MAIN_ROOT=/home/user/Keyword-Spotting-Alibaba
# Absolute path of compiled kaldi. See example below.
KALDI_ROOT=/home/user/kaldi
```
2. Set the backend method in `cmd.sh`. More details can be found in the file. (Default: slurm)

```sh
cmd_backend='slurm'
```

3. You can change the configuration in `run.sh`. More details can be found in the file.
4. Use the command below to run the training.

```sh
./run.sh
```
