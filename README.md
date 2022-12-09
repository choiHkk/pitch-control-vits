## Introduction
1. vits 오픈 소스를 활용하여 Pitch를 제어할 수 있는 모델을 구성하고, VCTK 데이터셋을 사용해 학습합니다.
2. VCTK 데이터셋은 기본적으로 48kHz인 점을 감안하여 22kHz로 resampling할 수 있도록 utils.load_wav_to_torch()를 수정했습니다.
3. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
4. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
5. VCTK 데이터셋은 ASR 모델을 활용한 Auto-preprocessing 과정을 거친 후 학습에 사용되었습니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/showmik50/vctk-dataset

## Docker build
1. git clone https://github.com/choiHkk/pitch-control-vits.git
2. `cd /path/to/the/pitch-control-vits`
3. `docker build --tag pitch-control-vits:latest .`

## Training
1. `nvidia-docker run -it --name 'pitch-control-vits' -v /path/to/pitch-control-vits:/home/work/pitch-control-vits --ipc=host --privileged pitch-control-vits:latest`
2. `cd /home/work/pitch-control-vits/monotonic_align`
3. `python setup.py build_ext --inplace`
4. `cd /home/work/pitch-control-vits`
5. `python train_ms.py -c ./config/vctk_base_hifigan.json -m vctk_base_hifigan`
6. arguments
  * -c : comfig path
  * -m : model output directory
7. (OPTIONAL) `tensorboard --logdir=outdir/logdir`


## Tensorboard losses
![pitch-control-vits-tensorboard-losses1](https://user-images.githubusercontent.com/69423543/184790465-ac09988c-1685-4f6d-b3c7-a5458596d348.png)
![pitch-control-vits-tensorboard-losses2](https://user-images.githubusercontent.com/69423543/184790469-60a181fb-1d79-4bc7-bac8-caf443871f78.png)


## Tensorboard Stats
![pitch-control-vits-tensorboard-stats](https://user-images.githubusercontent.com/69423543/184790478-551f543b-f002-40ba-bd08-a36ce07277c8.png)


## Reference
1. [VITS](https://arxiv.org/abs/2106.06103)
2. [VISinger](https://arxiv.org/abs/2110.08813)
3. [Period-VITS](https://arxiv.org/abs/2210.15964)
4. [FastSpeech2](https://arxiv.org/abs/2006.04558)
5. [HiFi-GAN](https://arxiv.org/abs/2010.05646)
