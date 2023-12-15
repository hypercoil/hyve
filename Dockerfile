FROM python:3.11
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN wget https://www.vtk.org/files/release/9.3/vtk-9.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install vtk-9.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install hyve[test]==0.0.2
RUN pip install hyve-examples
