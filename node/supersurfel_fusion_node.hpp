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


#ifndef SUPERSURFEL_FUSION_NODE_HPP
#define SUPERSURFEL_FUSION_NODE_HPP

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <supersurfel_fusion/supersurfel_fusion.hpp>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <tf/transform_listener.h>


namespace supersurfel_fusion
{

class SupersurfelFusionNode
{

public:
    SupersurfelFusionNode();
    ~SupersurfelFusionNode();
    void RGBDCallback(const sensor_msgs::ImageConstPtr& msg_rgb, const sensor_msgs::ImageConstPtr& msg_depth);
    void camInfoCallback(const sensor_msgs::CameraInfo& info);
    void publishModelMarker(const std_msgs::Header& header);
    void publishFrameMarker(const std_msgs::Header& header);
    void publishDefGraph(const std_msgs::Header& header);
    void publishDefConstraints(const std_msgs::Header& header);
    void run();

    void publishLocalMapPoints(const std_msgs::Header& header);
    void publishCenters(const std_msgs::Header& header);

private:
    ros::NodeHandle nh, privateNh;

    SupersurfelFusion ssf;

    float depthScale;

    bool display, saveModel;
    std::string modelFile;
    float confThresh;

    image_transport::ImageTransport it;
    image_transport::SubscriberFilter rgbSub, depthSub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RGBDSyncPolicy;
    message_filters::Synchronizer<RGBDSyncPolicy> sync;
    ros::Subscriber camInfoSub;

    std::string mapFrameId;
    ros::Publisher frameMarkerPub, modelMarkerPub, nodesPub, edgesPub, constraintsPub;
    tf::TransformBroadcaster br;
    nav_msgs::Path path;
    ros::Publisher pathPub;
    image_transport::Publisher slantedPlanePub, superpixelsPub, modMaskPub;
    sensor_msgs::ImagePtr slantedPlaneMsg, superpixelsMsg, modMaskMsg;

    ros::Publisher  odomPub;
    tf::Transform prevOptToMap;
    ros::Time prevTime;

    ros::Publisher localMapPub, centersPub;
};

} // namespace supersurfel_fusion

#endif // SUPERSURFEL_FUSION_NODE_HPP
