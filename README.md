# ElxaTST

> **E**fficient **L**ead-lag Cross(**X**) **A**ttention **T**ime **S**eries **T**ransformer

![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.2_LTS-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

## Reproducing

### Install dependencies

```
$ conda create -n tslib python=3.8
$ conda activate tslib
$ pip3 install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
$ pip install -e .
```

### Training models

```
$ ./scripts/long_term_forecast/ETT_script/ElxaTST_ETTh1_96_96.sh
$ ./scripts/long_term_forecast/ETT_script/ElxaTST_ETTh2_96_96.sh
$ ./scripts/long_term_forecast/ETT_script/ElxaTST_ETTm1_96_96.sh
$ ./scripts/long_term_forecast/ETT_script/ElxaTST_ETTm2_96_96.sh
$ ./scripts/long_term_forecast/Weather_script/ElxaTST_Weather_96_96.sh
```
