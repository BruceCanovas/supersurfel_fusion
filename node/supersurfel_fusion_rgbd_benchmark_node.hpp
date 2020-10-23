/*************************************************************************
* Copyright (C) 2020, Bruce Canovas, Amaury Negre, GIPSA-lab
* This file is part of https://github.com/BruceCanovas/supersurfel_fusion

* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*************************************************************************/


#ifndef SUPERSURFEL_FUSION_RGBD_BENCHMARK_NODE_HPP
#define SUPERSURFEL_FUSION_RGBD_BENCHMARK_NODE_HPP

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <supersurfel_fusion/supersurfel_fusion.hpp>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>
#include <dynamic_reconfigure/server.h>
#include <supersurfel_fusion/SupersurfelFusionRGBDBenchmarkConfig.h>


namespace supersurfel_fusion
{

class SupersurfelFusionRGBDBenchmarkNode
{

public:
    SupersurfelFusionRGBDBenchmarkNode();
    ~SupersurfelFusionRGBDBenchmarkNode();
    void publishModelMarker();
    void publishFrameMarker();
    void publishDefGraph();
    void publishDefConstraints();
    void run();
    void publishLocalMapPoints();
    void configCallback(supersurfel_fusion::SupersurfelFusionRGBDBenchmarkConfig &config, uint32_t level);

private:
    ros::NodeHandle nh, privateNh;

    SupersurfelFusion ssf;

    float depthScale;

    bool display, saveModel;
    std::string modelFile;
    float confThresh;

    image_transport::ImageTransport it;

    std::string mapFrameId, camFrameId;
    ros::Publisher frameMarkerPub, modelMarkerPub, nodesPub, edgesPub, constraintsPub;
    tf::TransformBroadcaster br;
    nav_msgs::Path path, gtPath;
    ros::Publisher pathPub, gtPathPub;
    image_transport::Publisher slantedPlanePub, superpixelsPub, modMaskPub;
    sensor_msgs::ImagePtr slantedPlaneMsg, superpixelsMsg, modMaskMsg;

    ros::Publisher  odomPub;
    tf::Transform prevOptToMap;
    ros::Time prevTime;

    ros::Publisher localMapPub;

    tf::Transform firstGtPose;

    std::string trajectoryFilePath, datasetPath;

    int cpt;

    bool stop;
};

} // namespace supersurfel_fusion

#endif // SUPERSURFEL_FUSION_RGBD_BENCHMARK_NODE_HPP
