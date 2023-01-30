FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

COPY ./requirements.txt /tmp/requirements.txt
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && \
    apt-get -y -qq update && \
    apt-get install -y apt-utils && \
    apt-get install -y curl && \
    apt-get install -y tzdata && \
    apt-get install -y python3-ipykernel && \
    apt-get install -y espeak && \
    apt-get install -y openssh-client && \
    apt-get install -y curl && \
    apt-get install -y git && \
    apt-get install -y vim && \
    apt-get install -y libsndfile-dev && \
    apt-get install -y gcc && \
    apt-get install -y ffmpeg && \
    apt-get install -y locales && \
    apt-get install -y language-pack-ko && \
    apt-get install -y default-jre && \
    apt-get install -y screen && \
    apt-get install -y zip && \
    apt-get install -y unzip && \
    apt-get install -y sshfs

RUN pip install --upgrade pip
RUN /bin/bash -c "bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh) &&\
                  pip install -r /tmp/requirements.txt --ignore-installed"
RUN pip install --upgrade tensorboard
RUN pip install jupyter -U

VOLUME /home/work
WORKDIR /home/work

CMD ["/bin/bash"]
