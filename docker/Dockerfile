FROM python:3.8.12

# 初始设置
ARG PROJECT_ROOT=/root/asr
WORKDIR ${PROJECT_ROOT}

# 拷贝文件
COPY ./requirements.txt ./requirements.txt

# RUN apt-get update && apt-get install -y gnupg
# 添加公钥
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6ED0E7B82643E131

# 安装所需模块
RUN echo 'deb http://mirrors.aliyun.com/debian/ bullseye main non-free contrib\n\
    deb https://mirrors.aliyun.com/debian-security/ bullseye-security main\n\
    deb https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib\n\
    deb https://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib'\
    > /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y git libgl1-mesa-glx

RUN apt-get -y install ffmpeg

COPY ./torch-2.2.1+cu118-cp38-cp38-linux_x86_64.whl ${PROJECT_ROOT}/
COPY ./torchaudio-2.2.1+cu118-cp38-cp38-linux_x86_64.whl ${PROJECT_ROOT}/
COPY ./nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl ${PROJECT_ROOT}/
COPY ./nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux1_x86_64.whl ${PROJECT_ROOT}/
COPY . ${PROJECT_ROOT}/

# 构建Python环境
RUN pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple \
    && pip install --upgrade pip \
    && pip install ./nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux1_x86_64.whl \
    && pip install ./nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl \
    && pip install ./torch-2.2.1+cu118-cp38-cp38-linux_x86_64.whl \
    && pip install ./torchaudio-2.2.1+cu118-cp38-cp38-linux_x86_64.whl

RUN pip install -r requirements.txt

RUN pip install -e ./ 
    

# Runtime paths, overridable with `docker run -e NAME=value`.
ENV MODELSCOPE_CACHE="${PROJECT_ROOT}/models_from_modelscope" \
    ITN_CACHE_DIR="${PROJECT_ROOT}/WeTextProcessing/itn"