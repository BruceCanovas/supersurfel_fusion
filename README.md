# SupersurfelFusion #

Dense RGB-D SLAM system RGB-D SLAM articulated around a supersurfel-based 3D
representation for fast, lightweight and compact mapping in indoor environment.

Check out the video:

[![SupersurfelFusion video](http://img.youtube.com/vi/hzzVVHUAO74/0.jpg)](https://www.youtube.com/watch?v=hzzVVHUAO74 "SupersurfelFusion video")

## Related Publications ##

Please cite this publication when using SupersurfelFusion: +TODO

Note that the software is experimental and some changes have been done since the publication of the paper. For instance in the part performing moving object detection and removal, we add the possibility to use a lightweight deep learning based object detection to extract humans and improve the robustness against dynamic elements. 

## Requirements ##

- ### Ubuntu 16.04 or 18.04 ###
The system has been tested both on Ubuntu 16.04 and 18.04.

- ### [ROS](https://www.ros.org/) Kinetic or Melodic ###
Use ROS Kinetic on Ubuntu 16.04 and Melodic on 18.04.

- ### [CUDA](https://developer.nvidia.com/cuda-downloads) >= 9.0 ###
The system has been tested on platforms with CUDA 9.0 and CUDA 10.2.

- ### [OpenCV](https://opencv.org/) 3.4 ###
We use OpenCV 3.4 but any version of OpenCV 3 should work. OpenCV has to be installed with CUDA support and with contrib modules.
Note that, to be able to use with ROS a version of OpenCV different from the default ROS OpenCV version (which doesn't have CUDA support), you might have to rebuild all ROS packages that require OpenCV against your specific version, particularly the [vision_opencv](http://wiki.ros.org/vision_opencv) package that provides [cv_bridge](http://wiki.ros.org/cv_bridge). If you have differents OpenCV versions installed on your computer, you can specify the one you want to use at build time by calling `catkin_make` like this:
```
$ catkin_make -DOpenCV_DIR=<your OpenCV path>
```
or by using this:
```
find_package(OpenCV REQUIRED
   NO_MODULE #Should be optional, tells CMake to use config mode
   PATHS  <your OpenCV path># Tells CMake to look here
   NO_DEFAULT_PATH #and don't look anywhere else)
```
inside your CMakeLists.txt instead of:
```
find_package(OpenCV 3 REQUIRED)
```
to be sure to link against the desired version.

- ### [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) ###
This dependency can be resolved by installing the following package: libsuitesparse-dev.

- ### [G2O](https://github.com/RainerKuemmerle/g2o) ###

## Third Parties ##

Our implementation integrates parts of codes from external libraries.

### [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) ###
For features extraction and associated descriptors computation.

### [GMS-Feature-Matcher](https://github.com/JiawangBian/GMS-Feature-Matcher) ###
We use Grid-based Motion Statistics for robust feature correspondence.

### [ElasticFusion](https://github.com/mp3guy/ElasticFusion) ###
We credit ElasticFusion as a significant basis for our deformation graph and loop closure implementation.

### [Lightweight Visual Tracking (LVT)](https://github.com/SAR-Research-Lab/lvt) ###
We based the design of our feature-based visual odometry on their method and our Perspective-n-Point solver has been implemented following their code.

### [YOLOv4](https://github.com/AlexeyAB/darknet) ###
We integrated tiny YOLOv4 in our pipeline for improving robustness in dynamic scenes by detecting persons. To see how to enable or disable its use please refer 
to [this section](#node).

## Licenses ##
SupersurfelFusion is released under a GPLv3 license (see `supersurfel_fusion/licenses/LICENSE-SupersurfelFusion.txt`).

SupersurfelFusion includes differents third-party open-source software, which themselves include third-party open-source software. Each of these components have their own license.

You can find the licenses in the repository `supersurfel_fusion/licenses/`.

## Build SupersurfelFusion ##

The system is provided as a ROS package which can be copied or cloned into your workspace and built directly using catkin.

### 1. Clone the repository ###

```
$ cd ~/catkin_ws/src
$ git clone https://gricad-gitlab.univ-grenoble-alpes.fr/canovasb/supersurfel_fusion.git
```

### 2. Build ###
First install darknet to use the YOLOv4 object detector using `make` in the darknet repository (`supersurfel_fusion/third_party/darknet`).
In the `Makefile` set:

- `GPU=1` to build enabling CUDA support (OPTIONAL)
- `OPENCV=1` to build with OpenCV
- `LIBSO=1` to build the library `darknet.so`

Once darknet installed just go to your catkin workspace root directory and build using `catkin_make`.

```
$ cd ~/catkin_ws
$ catkin_make
```

## Usage ##

Our system takes as input **registered** RGB-D frames. It is interfaced with a ROS node.

### Run with Intel Realsense ###

We provide a lauch file to start our system along an Intel RealSense cameras (D400 series SR300 camera and T265 Tracking Module) and an Rviz GUI. ROS drivers for Intel Realsense Devices can be found [here](https://github.com/IntelRealSense/realsense-ros). To use SupersurfelFusion, open a terminal and execute:
```
$ roslaunch supersurfel_fusion supersurfel_fusion_realsense_rviz.launch
```

### Run with rosbag or other RGB-D sensors ###

To use SupersurfelFusion with other devices you just need to remap the `/camera_info`, `/image_color` and `/image_depth` topics in the supersurfel_fusion_rviz.launch to the topics published by your sensor or bagfile. Then start your device on ROS with registered RGB-D stream, or play your rosbag and execute:
```
$ roslaunch supersurfel_fusion supersurfel_fusion_rviz.launch
```

### Run without RViz GUI ###

```
$ roslaunch supersurfel_fusion supersurfel_fusion_realsense.launch
```
or 
```
$ roslaunch supersurfel_fusion supersurfel_fusion.launch
```

### Run with [TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset) sequences  ###

```
$ roslaunch supersurfel_fusion supersurfel_fusion_rgbd_benchmark.launch
```
We provide two sequences in `supersurfel_fusion/rgbd_benchmark` that can be specified  in the `supersurfel_fusion_rgbd_benchmark.launch` launch file. When using this launch file, SupersurfelFusion processes every frame and can be played/paused anytime by checking the "stop" boxe of the rqt_reconfigure window that popped up. Estimated and ground truth trajectories are displayed in Rviz and the estimation is saved (location can be specified in the launch file) so it can be used for evaluation with [tools](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation) provided by the TUM. 

### Node ###

The `supersurfel_fusion_node`, executed by the differents launchfile, is the node that allows to run SupersurfelFusion under ROS.
There are differents parameters you can play with in the different launch files. For instance you can enable/disable the loop closure, enable/disable the moving object detection with or without YOLO... 

#### Subscribed topics: ####

- `\image_color`
- `\image_depth`
- `\camera_info`

#### Published topics: ####

- `\superpixels`: image of superpixels
- `\slanted_plane`: image of slanted plane associated to superpixels
- `\mod_mask`: binary mask of detected moving elements
- `\model_supersurfel_marker`: global map
- `\frame_supersurfel_marker`: frame supersurfels
- `\nodes_marker`: nodes of the deformation graph
- `\edges_marker`: edges of the deformation graph
- `\constraints_marker`: deformation constraints
- `\trajectory`: camera path
- `\vo`: camera odometry 
- `\local_map`: sparse vo local map point cloud

## TODO ##

* Use cuda streams when possible
* Minimize thread divergence
* Edit superpixel code to use reduction instead of atomic operations
* Optimize for pointer aliasing
