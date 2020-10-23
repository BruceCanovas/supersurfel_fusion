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


#include "supersurfel_fusion_node.hpp"
#include <iostream>
#include <tf/transform_datatypes.h>
#include <math.h>
#include <cv_bridge/cv_bridge.h>
#include <thrust/host_vector.h>
#include <nav_msgs/Odometry.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>


namespace supersurfel_fusion
{

SupersurfelFusionNode::SupersurfelFusionNode()
    : privateNh("~"),
      it(nh),
      rgbSub(it, "/image_color", 1),
      depthSub(it, "/image_depth", 1),
      sync(RGBDSyncPolicy(10), rgbSub, depthSub)
{
    privateNh.param("display", display, true);
    privateNh.param("map_frame_id", mapFrameId, std::string("map"));
    privateNh.param("save_model", saveModel, false);
    privateNh.param("model_file", modelFile, std::string("model.txt"));

    nodesPub = nh.advertise<visualization_msgs::Marker>("nodes_marker", 1);
    edgesPub = nh.advertise<visualization_msgs::Marker>("edges_marker", 1);
    constraintsPub = nh.advertise<visualization_msgs::MarkerArray>("constraints_marker", 1);
    frameMarkerPub = nh.advertise<visualization_msgs::Marker>("frame_supersurfel_marker", 1);
    modelMarkerPub = nh.advertise<visualization_msgs::Marker>("model_supersurfel_marker", 1);
    pathPub = nh.advertise<nav_msgs::Path>("trajectory", 1);

    slantedPlanePub = it.advertise("slanted_plane", 1);
    superpixelsPub = it.advertise("superpixels", 1);
    modMaskPub = it.advertise("mod_mask", 1);

    camInfoSub = nh.subscribe("/camera_info", 1, &SupersurfelFusionNode::camInfoCallback, this);

    odomPub = nh.advertise<nav_msgs::Odometry>("vo", 1);

    sync.registerCallback(boost::bind(&SupersurfelFusionNode::RGBDCallback, this, _1, _2));

    prevOptToMap.setIdentity();

    localMapPub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("local_map", 1);
}

SupersurfelFusionNode::~SupersurfelFusionNode()
{
}

void SupersurfelFusionNode::RGBDCallback(const sensor_msgs::ImageConstPtr& msg_rgb, const sensor_msgs::ImageConstPtr& msg_depth)
{
    if(ssf.isInitialized())
    {
        cv_bridge::CvImageConstPtr cv_ptr_rgb = cv_bridge::toCvShare(msg_rgb);
        cv_bridge::CvImageConstPtr cv_ptr_depth = cv_bridge::toCvShare(msg_depth);

        cv::Mat rgb = cv_ptr_rgb->image.clone();
        cv::Mat depth = cv_ptr_depth->image.clone();
        depth.convertTo(depth, CV_32FC1, depthScale);

        ssf.processFrame(rgb, depth);

        Transform3 pose = ssf.getPose();
        tf::Transform opt_to_map(tf::Matrix3x3(pose.R.rows[0].x, pose.R.rows[0].y, pose.R.rows[0].z,
                                               pose.R.rows[1].x, pose.R.rows[1].y, pose.R.rows[1].z,
                                               pose.R.rows[2].x, pose.R.rows[2].y, pose.R.rows[2].z),
                                 tf::Vector3(pose.t.x, pose.t.y, pose.t.z));


        tf::StampedTransform stamped_opt_to_map(opt_to_map, ros::Time::now(), mapFrameId, msg_rgb->header.frame_id);
        br.sendTransform(stamped_opt_to_map);

        path.header = msg_rgb->header;
        path.header.frame_id = mapFrameId;
        geometry_msgs::PoseStamped pose_stamped_msg;
        tf::poseStampedTFToMsg(tf::Stamped<tf::Transform>(opt_to_map, ros::Time::now(), mapFrameId), pose_stamped_msg);
        path.poses.push_back(pose_stamped_msg);

        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = mapFrameId;
        odom_msg.child_frame_id = msg_rgb->header.frame_id;
        tf::poseTFToMsg (opt_to_map, odom_msg.pose.pose);

        if(ssf.getStamp() > 1)
        {
            tf::Transform delta_base = prevOptToMap.inverse() * opt_to_map;
            double delta_t = (msg_rgb->header.stamp - prevTime).toSec();

            odom_msg.twist.twist.linear.x = delta_base.getOrigin().getX() / delta_t;
            odom_msg.twist.twist.linear.y = delta_base.getOrigin().getY() / delta_t;
            odom_msg.twist.twist.linear.z = delta_base.getOrigin().getZ() / delta_t;
            tf::Quaternion delta_rot = delta_base.getRotation();
            tfScalar angle = delta_rot.getAngle();
            tf::Vector3 axis = delta_rot.getAxis();
            tf::Vector3 angular_twist = axis * angle / delta_t;
            odom_msg.twist.twist.angular.x = angular_twist.x();
            odom_msg.twist.twist.angular.y = angular_twist.y();
            odom_msg.twist.twist.angular.z = angular_twist.z();
        }

        //odometry_msg.pose.covariance = ;
        //odometry_msg.twist.covariance = ;

        prevTime = msg_rgb->header.stamp;
        prevOptToMap = opt_to_map;

        if(odomPub.getNumSubscribers() > 0)
            odomPub.publish(odom_msg);

        if(pathPub.getNumSubscribers() > 0)
            pathPub.publish(path);

        if(modelMarkerPub.getNumSubscribers() > 0)
            publishModelMarker(msg_rgb->header);

        if(frameMarkerPub.getNumSubscribers() > 0)
            publishFrameMarker(msg_rgb->header);

        if(superpixelsPub.getNumSubscribers() > 0)
        {
            cv::Mat im_seg;
            ssf.computeSuperpixelSegIm(im_seg);
            superpixelsMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", im_seg).toImageMsg();
            superpixelsPub.publish(superpixelsMsg);
        }

        if(slantedPlanePub.getNumSubscribers() > 0)
        {
            cv::Mat im_slanted_plane;
            ssf.computeSlantedPlaneIm(im_slanted_plane);
            im_slanted_plane.convertTo(im_slanted_plane, CV_8U, 128);
            slantedPlaneMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_slanted_plane).toImageMsg();
            slantedPlanePub.publish(slantedPlaneMsg);
        }

        if(modMaskPub.getNumSubscribers() > 0)
        {
            cv::Mat im_mask;
            im_mask = ssf.getModMask().clone();
            modMaskMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_mask).toImageMsg();
            modMaskPub.publish(modMaskMsg);
        }

        if(nodesPub.getNumSubscribers() > 0 || edgesPub.getNumSubscribers() > 0)
            publishDefGraph(msg_rgb->header);

        if(constraintsPub.getNumSubscribers() > 0)
            publishDefConstraints(msg_rgb->header);

        if(localMapPub.getNumSubscribers() > 0)
            publishLocalMapPoints(msg_rgb->header);

        if(display)
        {
            cv::Mat im_seg, im_slanted_plane;
            ssf.computeSuperpixelSegIm(im_seg);
            ssf.computeSlantedPlaneIm(im_slanted_plane);
            cv::imshow("superpixels", im_seg);
            cv::imshow("slanted planes", im_slanted_plane * 0.3f);
            cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
            cv::imshow("rgb", rgb);
            cv::imshow("depth", depth * 0.3f);

            cv::waitKey(1);
        }
    }
}

void SupersurfelFusionNode::camInfoCallback(const sensor_msgs::CameraInfo& msg_cam_info)
{
    CamParam cam_param;
    cam_param.fx = float(msg_cam_info.K[0]);
    cam_param.cx = float(msg_cam_info.K[2]);
    cam_param.fy = float(msg_cam_info.K[4]);
    cam_param.cy = float(msg_cam_info.K[5]);
    cam_param.height = msg_cam_info.height;
    cam_param.width = msg_cam_info.width;

    int nb_supersurfels_max;
    int cell_size, seg_iter, nb_samples, filter_iter;
    float lambda_pos, lambda_bound, lambda_size, lambda_disp, thresh_disp;
    float filter_alpha, filter_beta, filter_threshold;
    bool seg_use_ransac;
    float conf_thresh_scale;
    int icp_iter;
    int delta_t;
    double icp_cov_thresh;
    float range_min, range_max;
    int nb_features, features_nb_levels, ini_th_fast, min_th_fast;
    float features_scale_factor;
    int untracked_threshold;
    bool enable_loop_closure, enable_mod;
    bool use_yolo;

    privateNh.param("cell_size", cell_size, 16);
    privateNh.param("seg_iter", seg_iter, 10);
    privateNh.param("icp_iter", icp_iter, 10);
    privateNh.param("icp_cov_thresh", icp_cov_thresh, 0.04);
    privateNh.param("lambda_pos", lambda_pos, 50.0f);
    privateNh.param("lambda_bound", lambda_bound, 1000.0f);
    privateNh.param("lambda_size", lambda_size, 10000.0f);
    privateNh.param("lambda_disp", lambda_disp, 1000000.0f);
    privateNh.param("thresh_disp", thresh_disp, 0.0001f);
    privateNh.param("seg_use_ransac", seg_use_ransac, true);
    privateNh.param("nb_samples", nb_samples, 16);
    privateNh.param("nb_supersurfels_max", nb_supersurfels_max, 50000);
    privateNh.param("filter_iter", filter_iter, 4);
    privateNh.param("filter_alpha", filter_alpha, 0.1f);
    privateNh.param("filter_beta", filter_beta, 1.0f);
    privateNh.param("filter_threshold", filter_threshold, 0.05f);
    privateNh.param("conf_thresh_scale", conf_thresh_scale, 5.0f);
    privateNh.param("delta_t", delta_t, 10);
    privateNh.param("range_min", range_min, 0.2f);
    privateNh.param("range_max", range_max, 5.0f);
    privateNh.param("nb_features", nb_features, 2000);
    privateNh.param("features_scale_factor", features_scale_factor, 1.2f);
    privateNh.param("features_nb_levels", features_nb_levels, 8);
    privateNh.param("ini_th_fast", ini_th_fast, 20);
    privateNh.param("min_th_fast", min_th_fast, 5);
    privateNh.param("untracked_threshold", untracked_threshold, 10);
    privateNh.param("enable_loop_closure", enable_loop_closure, true);
    privateNh.param("depth_scale", depthScale, 0.001f);
    privateNh.param("enable_mod", enable_mod, true);

    confThresh = cell_size * cell_size * conf_thresh_scale;

    ssf.initialize(cam_param,
                   cell_size,
                   lambda_pos,
                   lambda_bound,
                   lambda_size,
                   lambda_disp,
                   thresh_disp,
                   seg_iter,
                   seg_use_ransac,
                   nb_samples,
                   filter_iter,
                   filter_alpha,
                   filter_beta,
                   filter_threshold,
                   range_min,
                   range_max,
                   delta_t,
                   confThresh,
                   nb_supersurfels_max,
                   icp_iter,
                   icp_cov_thresh,
                   nb_features,
                   features_scale_factor,
                   features_nb_levels,
                   ini_th_fast,
                   min_th_fast,
                   untracked_threshold,
                   enable_loop_closure,
                   enable_mod);

    privateNh.param("use_yolo", use_yolo,false);

    if(use_yolo)
    {
        std::string yolo_names_file, yolo_weights_file, yolo_config_file;

        privateNh.getParam("yolo_names_file", yolo_names_file);
        privateNh.getParam("yolo_config_file", yolo_config_file);
        privateNh.getParam("yolo_weights_file", yolo_weights_file);

        ssf.setYoloDetector(yolo_config_file, yolo_weights_file, yolo_names_file);
    }

    camInfoSub.shutdown();

    std::cout<<"Camera info received"<<std::endl;
}

void SupersurfelFusionNode::publishModelMarker(const std_msgs::Header& header)
{
    thrust::host_vector<float3> positions(ssf.getModel().positions);
    thrust::host_vector<float3> colors(ssf.getModel().colors);
    thrust::host_vector<Mat33> orientations(ssf.getModel().orientations);
    thrust::host_vector<float2> dims(ssf.getModel().dims);
    thrust::host_vector<float> confidences(ssf.getModel().confidences);

    visualization_msgs::Marker model_marker_msg;

    model_marker_msg.header.stamp = header.stamp;
    model_marker_msg.header.frame_id = mapFrameId;

    model_marker_msg.type = visualization_msgs::Marker::TRIANGLE_LIST;
    model_marker_msg.action = 0;

    tf::Transform pose = tf::Transform(tf::Quaternion(0,0,0,1),
                                       tf::Vector3(0,0,0));
    tf::poseTFToMsg(pose, model_marker_msg.pose);

    model_marker_msg.scale.x = 1;
    model_marker_msg.scale.y = 1;
    model_marker_msg.scale.z = 1;

    model_marker_msg.points.resize(positions.size() * 6);
    model_marker_msg.colors.resize(colors.size() * 6);

    float thresh = confThresh;

    for(size_t i = 0; i < positions.size(); ++i)
    {
        if(confidences[i] > thresh)
        {
            float v0 = 3.0f * sqrtf(dims[i].x);
            float v1 = 3.0f * sqrtf(dims[i].y);
            float3 e0 = orientations[i].rows[0];
            float3 e1 = orientations[i].rows[1];

            float3 pos = positions[i];

            if(!std::isfinite(v0))
                v0 = 0;
            if(!std::isfinite(v1))
                v1 = 0;

            if(!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z))
            {
                pos.x = 0.f;
                pos.y = 0.f;
                pos.z = 0.f;
            }

            tf::Point p0(pos.x + v0 * e0.x + v1 * e1.x,
                         pos.y + v0 * e0.y + v1 * e1.y,
                         pos.z + v0 * e0.z + v1 * e1.z);

            tf::Point p1(pos.x + v0 * e0.x - v1 * e1.x,
                         pos.y + v0 * e0.y - v1 * e1.y,
                         pos.z + v0 * e0.z - v1 * e1.z);

            tf::Point p2(pos.x - v0 * e0.x - v1 * e1.x,
                         pos.y - v0 * e0.y - v1 * e1.y,
                         pos.z - v0 * e0.z - v1 * e1.z);

            tf::Point p3(pos.x - v0 * e0.x + v1 * e1.x,
                         pos.y - v0 * e0.y + v1 * e1.y,
                         pos.z - v0 * e0.z + v1 * e1.z);

            tf::pointTFToMsg(p0, model_marker_msg.points[6*i]);
            tf::pointTFToMsg(p1, model_marker_msg.points[6*i+1]);
            tf::pointTFToMsg(p2, model_marker_msg.points[6*i+2]);
            tf::pointTFToMsg(p0, model_marker_msg.points[6*i+3]);
            tf::pointTFToMsg(p2, model_marker_msg.points[6*i+4]);
            tf::pointTFToMsg(p3, model_marker_msg.points[6*i+5]);

            std_msgs::ColorRGBA color;
            color.r = colors[i].x / 255;
            color.g = colors[i].y / 255;
            color.b = colors[i].z / 255;
            color.a = 1.;

            model_marker_msg.colors[6*i] = color;
            model_marker_msg.colors[6*i+1] = color;
            model_marker_msg.colors[6*i+2] = color;
            model_marker_msg.colors[6*i+3] = color;
            model_marker_msg.colors[6*i+4] = color;
            model_marker_msg.colors[6*i+5] = color;

        }
        else
        {
            geometry_msgs::Point p;
            p.x = p.y = p.z = 0;
            model_marker_msg.points[6*i]=p;
            model_marker_msg.points[6*i+1]=p;
            model_marker_msg.points[6*i+2]=p;
            model_marker_msg.points[6*i+3]=p;
            model_marker_msg.points[6*i+4]=p;
            model_marker_msg.points[6*i+5]=p;
            std_msgs::ColorRGBA color;
            color.r = 0;
            color.g = 0;
            color.b = 0;
            color.a = 1.;
            model_marker_msg.colors[6*i] = color;
            model_marker_msg.colors[6*i+1] = color;
            model_marker_msg.colors[6*i+2] = color;
            model_marker_msg.colors[6*i+3] = color;
            model_marker_msg.colors[6*i+4] = color;
            model_marker_msg.colors[6*i+5] = color;
        }
    }

    modelMarkerPub.publish(model_marker_msg);
}

void SupersurfelFusionNode::publishFrameMarker(const std_msgs::Header& header)
{
    thrust::host_vector<float3> positions(ssf.getFrame().positions);
    thrust::host_vector<float3> colors(ssf.getFrame().colors);
    thrust::host_vector<Mat33> orientations(ssf.getFrame().orientations);
    thrust::host_vector<float2> dims(ssf.getFrame().dims);
    thrust::host_vector<float> confidences(ssf.getFrame().confidences);

    visualization_msgs::Marker frame_marker_msg;

    frame_marker_msg.header = header;
    frame_marker_msg.header.frame_id = header.frame_id;

    frame_marker_msg.type = visualization_msgs::Marker::TRIANGLE_LIST;
    frame_marker_msg.action = 0;

    frame_marker_msg.pose.position.x = 0;
    frame_marker_msg.pose.position.y = 0;
    frame_marker_msg.pose.position.z = 0;
    frame_marker_msg.pose.orientation.x = 0;
    frame_marker_msg.pose.orientation.y = 0;
    frame_marker_msg.pose.orientation.z = 0;
    frame_marker_msg.pose.orientation.w = 1;

    frame_marker_msg.scale.x = 1;
    frame_marker_msg.scale.y = 1;
    frame_marker_msg.scale.z = 1;

    frame_marker_msg.points.resize(positions.size() * 6);
    frame_marker_msg.colors.resize(colors.size() * 6);

    for(size_t i = 0; i < positions.size(); ++i)
    {
        if(confidences[i] > 0.0f)
        {
            float v0 = 3.0f * sqrtf(dims[i].x);
            float v1 = 3.0f * sqrtf(dims[i].y);
            float3 e0 = orientations[i].rows[0];
            float3 e1 = orientations[i].rows[1];

            float3 pos = positions[i];

            if(!std::isfinite(v0))
                v0 = 0;
            if(!std::isfinite(v1))
                v1 = 0;

            if(!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z))
            {
                pos.x = 0.f;
                pos.y = 0.f;
                pos.z = 0.f;
            }

            tf::Point p0(pos.x + v0 * e0.x + v1 * e1.x,
                         pos.y + v0 * e0.y + v1 * e1.y,
                         pos.z + v0 * e0.z + v1 * e1.z);
            tf::Point p1(pos.x + v0 * e0.x - v1 * e1.x,
                         pos.y + v0 * e0.y - v1 * e1.y,
                         pos.z + v0 * e0.z - v1 * e1.z);
            tf::Point p2(pos.x - v0 * e0.x - v1 * e1.x,
                         pos.y - v0 * e0.y - v1 * e1.y,
                         pos.z - v0 * e0.z - v1 * e1.z);
            tf::Point p3(pos.x - v0 * e0.x + v1 * e1.x,
                         pos.y - v0 * e0.y + v1 * e1.y,
                         pos.z - v0 * e0.z + v1 * e1.z);

            tf::pointTFToMsg(p0, frame_marker_msg.points[6*i]);
            tf::pointTFToMsg(p1, frame_marker_msg.points[6*i+1]);
            tf::pointTFToMsg(p2, frame_marker_msg.points[6*i+2]);
            tf::pointTFToMsg(p0, frame_marker_msg.points[6*i+3]);
            tf::pointTFToMsg(p2, frame_marker_msg.points[6*i+4]);
            tf::pointTFToMsg(p3, frame_marker_msg.points[6*i+5]);

            std_msgs::ColorRGBA color;
            color.r = colors[i].x / 255;
            color.g = colors[i].y / 255;
            color.b = colors[i].z / 255;
            color.a = 1.;
            frame_marker_msg.colors[6*i] = color;
            frame_marker_msg.colors[6*i+1] = color;
            frame_marker_msg.colors[6*i+2] = color;
            frame_marker_msg.colors[6*i+3] = color;
            frame_marker_msg.colors[6*i+4] = color;
            frame_marker_msg.colors[6*i+5] = color;
        }
        else
        {
            geometry_msgs::Point p;
            p.x = p.y = p.z = 0;
            frame_marker_msg.points[6*i]=p;
            frame_marker_msg.points[6*i+1]=p;
            frame_marker_msg.points[6*i+2]=p;
            frame_marker_msg.points[6*i+3]=p;
            frame_marker_msg.points[6*i+4]=p;
            frame_marker_msg.points[6*i+5]=p;
            std_msgs::ColorRGBA color;
            color.r = 0;
            color.g = 0;
            color.b = 0;
            color.a = 1.;
            frame_marker_msg.colors[6*i] = color;
            frame_marker_msg.colors[6*i+1] = color;
            frame_marker_msg.colors[6*i+2] = color;
            frame_marker_msg.colors[6*i+3] = color;
            frame_marker_msg.colors[6*i+4] = color;
            frame_marker_msg.colors[6*i+5] = color;
        }
    }

    frameMarkerPub.publish(frame_marker_msg);
}

void SupersurfelFusionNode::publishDefGraph(const std_msgs::Header& header)
{
    thrust::host_vector<float3> nodes_positions = ssf.getNodesPositions();
    thrust::host_vector<int4> nodes_neighbours =  ssf.getNodesNeighbours();

    visualization_msgs::Marker nodes_msg;
    nodes_msg.header.stamp = header.stamp;
    nodes_msg.header.frame_id = mapFrameId;
    nodes_msg.type = visualization_msgs::Marker::SPHERE_LIST;
    nodes_msg.action = 0;
    nodes_msg.pose.position.x = 0;
    nodes_msg.pose.position.y = 0;
    nodes_msg.pose.position.z = 0;
    nodes_msg.pose.orientation.x = 0;
    nodes_msg.pose.orientation.y = 0;
    nodes_msg.pose.orientation.z = 0;
    nodes_msg.pose.orientation.w = 1;
    nodes_msg.scale.x = 0.03;
    nodes_msg.scale.y = 0.03;
    nodes_msg.scale.z = 0.03;
    nodes_msg.color.r = 0.0f;
    nodes_msg.color.g = 1.0f;
    nodes_msg.color.b = 0.0f;
    nodes_msg.color.a = 1.0f;

    visualization_msgs::Marker edges_msg;
    edges_msg.header.stamp = header.stamp;
    edges_msg.header.frame_id = mapFrameId;
    edges_msg.type = visualization_msgs::Marker::LINE_LIST;
    edges_msg.action = 0;
    edges_msg.pose = nodes_msg.pose;
    edges_msg.scale.x = 0.001;
    edges_msg.scale.y = 0.001;
    edges_msg.scale.z = 0.001;
    edges_msg.color.r = 1.0f;
    edges_msg.color.g = 0.0f;
    edges_msg.color.b = 0.0f;
    edges_msg.color.a = 1.0f;

    for(size_t i=0; i< nodes_positions.size(); i++)
    {
        geometry_msgs::Point p, n1, n2, n3, n4;

        tf::Point p_tf(nodes_positions[i].x, nodes_positions[i].y, nodes_positions[i].z);
        tf::pointTFToMsg(p_tf, p);
        nodes_msg.points.push_back(p);

        edges_msg.points.push_back(p);
        tf::Point n1_tf(nodes_positions[nodes_neighbours[i].x].x,
                        nodes_positions[nodes_neighbours[i].x].y,
                        nodes_positions[nodes_neighbours[i].x].z);
        tf::pointTFToMsg(n1_tf, n1);
        edges_msg.points.push_back(n1);

        edges_msg.points.push_back(p);
        tf::Point n2_tf(nodes_positions[nodes_neighbours[i].y].x,
                        nodes_positions[nodes_neighbours[i].y].y,
                        nodes_positions[nodes_neighbours[i].y].z);
        tf::pointTFToMsg(n2_tf, n2);
        edges_msg.points.push_back(n2);

        edges_msg.points.push_back(p);
        tf::Point n3_tf(nodes_positions[nodes_neighbours[i].z].x,
                        nodes_positions[nodes_neighbours[i].z].y,
                        nodes_positions[nodes_neighbours[i].z].z);
        tf::pointTFToMsg(n3_tf, n3);
        edges_msg.points.push_back(n3);

        edges_msg.points.push_back(p);
        tf::Point n4_tf(nodes_positions[nodes_neighbours[i].w].x,
                        nodes_positions[nodes_neighbours[i].w].y,
                        nodes_positions[nodes_neighbours[i].w].z);
        tf::pointTFToMsg(n4_tf, n4);
        edges_msg.points.push_back(n4);
    }

    nodesPub.publish(nodes_msg);
    edgesPub.publish(edges_msg);
}

void SupersurfelFusionNode::publishDefConstraints(const std_msgs::Header& header)
{
    thrust::host_vector<float3> source_pts = ssf.getConstraintsSources();
    thrust::host_vector<float3> target_pts = ssf.getConstraintsTargets();

    visualization_msgs::MarkerArray arrow_array_msg;

    visualization_msgs::Marker arrow_msg;

    arrow_msg.header.stamp = header.stamp;
    arrow_msg.header.frame_id = mapFrameId;
    arrow_msg.action = 0;
    arrow_msg.type = visualization_msgs::Marker::ARROW;
    arrow_msg.color.r = 0.0f;
    arrow_msg.color.g = 0.0f;
    arrow_msg.color.b = 1.0f;
    arrow_msg.color.a = 1.0f;
    arrow_msg.scale.x = 0.01f;
    arrow_msg.scale.y = 0.03f;
    arrow_msg.scale.z = 0.0f;

    arrow_msg.ns = "graph_constraints";

    for(size_t i=0; i < source_pts.size(); i++)
    {
        if(!(source_pts[i].x == target_pts[i].x && source_pts[i].y == target_pts[i].y && source_pts[i].z == target_pts[i].z))
        {

            geometry_msgs::Point p_src, p_target;

            tf::Point p_src_tf(source_pts[i].x, source_pts[i].y, source_pts[i].z);
            tf::pointTFToMsg(p_src_tf, p_src);

            tf::Point p_target_tf(target_pts[i].x, target_pts[i].y, target_pts[i].z);
            tf::pointTFToMsg(p_target_tf, p_target);

            arrow_msg.points.clear();
            arrow_msg.points.push_back(p_src);
            arrow_msg.points.push_back(p_target);

            arrow_msg.id = i;

            arrow_array_msg.markers.push_back(arrow_msg);
        }
    }

    constraintsPub.publish(arrow_array_msg);
}

void SupersurfelFusionNode::publishLocalMapPoints(const std_msgs::Header& header)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_map_cloud(new  pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB point_rgb;

    for(size_t i = 0; i < ssf.getLocalMapPoints().size(); i++)
    {
        point_rgb.r = 255;
        point_rgb.g = 255;
        point_rgb.b = 255;

        point_rgb.x = ssf.getLocalMapPoints()[i](0);
        point_rgb.y = ssf.getLocalMapPoints()[i](1);
        point_rgb.z = ssf.getLocalMapPoints()[i](2);

        local_map_cloud->points.push_back(point_rgb);
    }

    local_map_cloud->header.frame_id =  mapFrameId;
    pcl_conversions::toPCL(header.stamp, local_map_cloud->header.stamp);
    localMapPub.publish(local_map_cloud);
}

void SupersurfelFusionNode::run()
{
    ros::spin();

    if(saveModel)
        ssf.exportModel(modelFile);
}

} // namespace supersurfel_fusion
