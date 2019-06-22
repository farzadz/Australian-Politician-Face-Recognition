FROM ubuntu:16.04

RUN apt-get update \
  && apt-get install -y python3-pip python3 \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ENV maintainer farzadz



# ADD ./requirements.txt ~/model/


# RUN apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
# RUN apt-get install -y libcairo2-dev


COPY requirements.txt /home/ubuntu/model/
WORKDIR /home/ubuntu/model
RUN pip3 install -r requirements.txt


RUN apt-get update && apt-get install -y \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libav-tools  \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libjasper-dev \ 
    python-numpy \ 
    python-pycurl \ 
    python-opencv

COPY . .

CMD python server.py
