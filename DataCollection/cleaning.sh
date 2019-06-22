#!/bin/sh

# install packages
sudo apt-get update
sudo apt install git
git clone https://github.com/farzadz/mtcnn-face.git
sudo apt install python3-pip
pip3 install tensorflow
pip3 install opencv-python
sudo apt-get install libsm6 libxrender1 libfontconfig1
pip3 install matplotlib
sudo apt-get install python3-tk
pip3 install Pillow
pip3 install tensorflow-gpu

# perform face extraction
# copy files in mtcnn-face-extraction folder to the parent
# directory of the directorys containing images first
for dir in *_*; do (python3 extract.py "$dir"); done;
# delete empty file
for dir in *; do (find "$dir" -size 0 -delete); done;

# install CUDA10.0
wget "https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux"
chmod +x cuda_10.0.130_410.48_linux.run
./cuda_10.0.130_410.48_linux.run --extract=$HOME
sudo ./cuda_10.0.130_410.48_linux.run
sudo vi /etc/environment # add :/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1 (including the ":") at the end of the PATH="/blah:/blah/blah" string (inside the quotes)
sudo reboot # can do a hard reboot here
cd /usr/local/cuda-10.1/samples/bin/x86_64/linux/release
./deviceQuery
nvidia-smi

# install cuDNN
# from https://developer.nvidia.com/rdp/cudnn-download, download all 3 .deb files: the runtime library, the developer library, and the code samples library
sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.5.0.56-1+cuda10.1_amd64.deb
cp -r /usr/src/cudnn_samples_v7/ ~
cd ~/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN

# install package for MobileNet
pip3 install keras

# install packages for clustering
pip3 install sklearn
pip3 install imutils

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-410
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0

# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update && \
        sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0

# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update && \
        sudo apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0

# perform clustering
# copy files in face-clustering folder to the parent
# directory of the directorys containing images first
for dir in *_*; do (python3 cluster.py "$dir"); done;