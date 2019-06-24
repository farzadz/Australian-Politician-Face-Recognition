# Australian-Politician-Face-Recognition

This repository contains the source code for COMP90055 project Australian Politician Face Recognition.

This project aims to develop a deep learning based system for recognition of the faces of the 225 members of the 45th Australian parliament. The outputs of the project contain an Android application for real-time face recognition of the targeted people using the mobile camera, and a Docker image that yields the same results in near real-time for all clients that request such service over a RESTful API. 


All the computing tasks in this project, including data collection, data pre-processing, and training the models, were performed on an instance from The National eResearch Collaboration Tools and Resources (NeCTAR). The instance with flavour uom.gpgpu.k80-6c55g, armed with a30GB disk, 6 vCPUs, 55GB of memory, and one general purpose NVIDIA Tesla K80 12GB GPU, further enhanced by attaching 60GB of volume storage, making itready for the large output of web crawls. The image running on this instance was NeCTAR Ubuntu18.04LTS (Bionic) amd64 and CUDA9.1 toolkit was installed over it for providing GPU acceleration with Tensorflow. Python3.6.7 was the programming language used for all developments except the Android application. All the models used in this project are based on Keras module of Tensorflow 1.13.1 and Tensorboard is used for the visualisation of the models’ outputs. Further details of packages and modules are accessible via the requirements.txt files. The Android application was developed using Java1.8 and Gradle5.1.1. The minimum supported SDK, and the target SDK versions are 21 and 26, respectively, with Google Mobile Service version 17.0.2.
### Environment Setup

To install all packages and GPU driver required for data processing and training, run setup.sh.

### Training

Please find related code under Training folder.

### Data Collection and Pre-processing

Please find related code under DataCollection folder.

### Web Service

Docker image is avaialable via dockerhub `farzadz/vggface:1.4` image. For more information check `Web Service` directory.

### Android Mobile Application

Please find related code under AndroidApplication folder.
