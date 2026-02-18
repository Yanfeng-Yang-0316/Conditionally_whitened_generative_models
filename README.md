# [Conditionally whitened generative models (CW-Gen)](https://openreview.net/forum?id=GG01lCopSK)

This repository contains the codes of *Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting*, accepted by ICLR 2026. CW-Gen is a framework that enhances probabilistic time series forecasting by incorporating conditional mean and covariance into generative models such as diffusion and flow matching. 

We first established a theorem to show when incorporating conditional mean and covariance is beneficial. Then, we proposed a Joint Mean-Covariance Estimator (JMCE) to estimate the conditional mean & cov. Finally, we propose a easy and general way to incorporate conditional mean & cov into diffusion model and flow matching.

## Environment
We adopt the [env of NsDiff](https://github.com/wwy155/NsDiff/blob/main/requirements.txt). There are some additional packages and you can just pip install them.

## Usage
First use 
```bash
from torch_timeseries.dataset import *
dataset = ETTh1(root='ts_datasets')
```
to download the datasets. Then, run the codes, e.g.:
```bash
python cw_diffusion_ts_e2e.py
```
I wrote the code really in a hurry, and I use LLM to translated comments. So, there might be some minor, simple bugs (like nonexistent path etc). These bugs are easy to fix, and don't effect the results.

## Acknowledgement
This project builds upon the implementation of [TimeDiff](https://arxiv.org/abs/2306.05043), [SSSD](https://github.com/AI4HealthUOL/SSSD), [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS), [TMDM](https://github.com/LiYuxin321/TMDM), [NsDiff](https://github.com/wwy155/NsDiff), and [FlowTS](https://github.com/UNITES-Lab/FlowTS). We specially thank the original authors of NsDiff.


## Reference
If you find the underlining method is helpful to your research, please cite:
```bibtex
@inproceedings{
yang_2026_cwgen,
title={Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting},
author={Yanfeng Yang and Siwei Chen and Pingping Hu and Zhaotong Shen and Yingjie Zhang and Zhuoran Sun and Shuai Li and Ziqi Chen and Kenji Fukumizu},
booktitle={International Conference on Learning Representations},
year={2026},
}
