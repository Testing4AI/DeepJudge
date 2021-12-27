# DeepJudge: Testing for Copyright Protection 
This repository contains code for paper [Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models](https://arxiv.org/abs/2112.05588) (IEEE S&P'22)

## Prerequisite (Py3 & TF2) 
The code are run successfully using **Python 3.6** and **Tensorflow 2.2.0.** 
We recommend using conda to install the tensorflow-gpu environment:
```shell
$ conda create -n tf2-gpu tensorflow-gpu==2.2.0
$ conda activate tf2-gpu
```

To run the code in jupyter notebook, you should add the kernel: 
```shell
$ pip install ipykernel
$ python -m ipykernel install --name tf2-gpu
```

## Files
- `DeepJudge`: model similarity metrics and test case generation methods.
- `train_models`: train clean models and suspect models.
- `watermarking-whitebox`: An TF2 implementation of [1]. (based on [repository](https://github.com/yu4u/dnn-watermark))
- `watermarking-blackbox`: An TF2 implementation of [2].
- `fingerprinting-blackbox`: An TF2 implementation of [3].

**Reference:** 
[1] Uchida, Yusuke, et al. "Embedding watermarks into deep neural networks." ICMR 2017.
[2] Zhang, Jialong, et al. "Protecting intellectual property of deep neural networks with watermarking." AisaCCS 2018.
[3] Cao, Xiaoyu, Jinyuan Jia, and Neil Zhenqiang Gong. "IPGuard: Protecting intellectual property of deep neural networks via fingerprinting the classification boundary." AsiaCCS 2021.

