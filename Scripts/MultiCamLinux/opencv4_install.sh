#!/bin/sh

# Check if the virtual environment path is provided
if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 VIRTUALENV_PATH" >&2
  exit 1
fi

# Configuration
HOME_DIR=$HOME
OPENCV_VERSION="4.9.0" # Replace with the latest version of OpenCV

# Automatically locate the Python executable and derive the version
PYTHON_EXECUTABLE=$(which python3)
PYTHON_VERSION=$($PYTHON_EXECUTABLE -c 'import platform; print(platform.python_version())')
PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

# Update and Upgrade the System
sudo apt-get update
sudo apt-get upgrade -y

# Install Dependencies
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y gstreamer1.0-tools
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran

# Download and Unpack OpenCV
cd ${HOME_DIR}
wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip -o opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
unzip -o opencv_contrib.zip

# Build and Install OpenCV
cd ${HOME_DIR}/opencv-${OPENCV_VERSION}/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${1} \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=${HOME_DIR}/opencv_contrib-${OPENCV_VERSION}/modules \
    -D PYTHON_DEFAULT_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -D PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
    -D WITH_OPENCV_PYTHON3=ON \
    -D WITH_OPENMP=ON \
    -D WITH_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=TRUE ..

make -j$(nproc) # Uses all available cores
sudo make install
sudo ldconfig

# Clean up
cd ${HOME_DIR}
rm -rf opencv-${OPENCV_VERSION} opencv_contrib-${OPENCV_VERSION} opencv.zip opencv_contrib.zip
