# gz sim layer
FROM ubuntu:24.04 as gz-base
ENV DEBIAN_FRONTEND=noninteractive
ENV GZ_SIM_RESOURCE_PATH=/workspace

RUN apt-get update && apt-get install -y \
  wget \
  gnupg \
  lsb-release

RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list

RUN apt-get update && apt-get install -y gz-harmonic

# build tools layer
FROM gz-base AS builder

RUN apt-get update && apt-get install -y \
  cmake \
  ninja-build \
  git \
  g++ \
  pkg-config \
  libzip-dev \
  libpugixml-dev

# FMI4cpp
WORKDIR /tmp
RUN git clone https://github.com/NTNU-IHB/FMI4cpp.git
WORKDIR /tmp/FMI4cpp
RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DFMI4CPP_BUILD_TESTS=OFF \
    -DFMI4CPP_BUILD_EXAMPLES=OFF
RUN cmake --build build
RUN cmake --install build

# runtime dev deps
FROM gz-base AS runtime

COPY --from=builder /usr/local /usr/local

RUN apt-get install -y software-properties-common \
  && add-apt-repository ppa:maveonair/helix-editor \
  && apt-get install -y vim \
    tmux \
    helix \
    clang \
    clang-tools \
    clangd \
    bash-completion

WORKDIR /workspace
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN echo 'PS1="🤖  \[\e[38;5;130m\]\u@\h \[\e[34m\]\w\[\e[0m\] $ "' >> /root/.bashrc
