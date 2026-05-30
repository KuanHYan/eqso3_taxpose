# FROM nvidia/cuda:12.2.1-base-ubuntu22.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Set up the environment.
ENV CODING_ROOT=/opt/pairpose
WORKDIR $CODING_ROOT

# Install the dependencies.
RUN apt-get update && apt-get install -y \
    # Dependencies required for python.
    build-essential \
    curl \
    ffmpeg \
    git \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    libreadline-dev \
    libxml2-dev \
    libxmlsec1-dev \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv.
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT=$CODING_ROOT/.pyenv
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install python.
RUN pyenv install 3.10.0
RUN pyenv global 3.10.0

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code
WORKDIR $CODING_ROOT/code

# Setup environment variables for NVIDIA and VirtualGL
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# Copy in the requirements.
COPY requirements-gpu.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir wheel==0.40.0

# Install the requirements.
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy in the third-party directory.
COPY third_party/dcp third_party/dcp
COPY third_party/vnn third_party/vnn

# # Install the third-party libraries.
# RUN pip install --no-cache-dir -e third_party/ndf_robot

# Install pyrep.
# RUN pip install --no-cache-dir --no-build-isolation "pyrep @ git+https://gitee.com/zhkhhust/PyRep.git"

# Copy in pyproject.toml.
COPY pyproject.toml .
RUN mkdir taxpose
RUN touch taxpose/py.typed

# Install our project.
RUN pip install --no-cache-dir -e ".[develop,rlbench]"

# Copy in the code.
COPY . .

# Make directories for mounting.
RUN mkdir $CODING_ROOT/data
RUN mkdir $CODING_ROOT/logs

COPY ./docker/entrypoint.sh /opt/pairpose/entrypoint.sh
ENTRYPOINT ["/opt/pairpose/entrypoint.sh"]

COPY Pointnet2_PyTorch/pointnet2_ops_lib
RUN cd Pointnet2_PyTorch/pointnet2_ops_lib && \
    pip install .

# {
#     "iptables": false,
#     "bridge": "none",
#     "ipv6": false,
#     "runtimes": {
#         "nvidia": {
#             "args": [],
#             "path": "nvidia-container-runtime"
#         }
#     },
#     "registry-mirrors": [
#         "https://dockerproxy.com",
#         "https://docker.m.daocloud.io",
#         "https://cr.console.aliyun.com",
#         "https://ccr.ccs.tencentyun.com",
#         "https://hub-mirror.c.163.com",
#         "https://mirror.baidubce.com",
#         "https://docker.nju.edu.cn",
#         "https://docker.mirrors.sjtug.sjtu.edu.cn",
#         "https://registry.docker-cn.com"
#     ]
# }