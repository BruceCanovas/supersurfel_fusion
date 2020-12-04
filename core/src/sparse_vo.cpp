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


#include <supersurfel_fusion/sparse_vo.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


namespace supersurfel_fusion
{

SparseVO::SparseVO()
{
    orb = new ORB_SLAM2::ORBextractor(1000, 1.2f, 8, 20, 7);

    rangeMin = 0.2f;
    rangeMax = 5.0f;

    cam.fx = 0.0f;
    cam.fy = 0.0f;
    cam.cx = 0.0f;
    cam.cy = 0.0f;
    cam.width = -1;
    cam.height = -1;

    pose.setIdentity();

    prevNbMatch = 0;

//    lastOk = false;

//    lastPosition.setZero();
//    linearVelocity.setZero();
//    lastQuaternion = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
//    angularVelocity = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
}

SparseVO::SparseVO(int nb_features,
                   float scale_factor,
                   int nb_levels,
                   int ini_th_fast,
                   int min_th_fast,
                   const CamParam& cam_param,
                   float range_min,
                   float range_max,
                   int untracked_threshold)
{
    orb = new ORB_SLAM2::ORBextractor(nb_features, scale_factor, nb_levels, ini_th_fast, min_th_fast);

    cam = cam_param;

    rangeMin = range_min;
    rangeMax = range_max;

    localMap = new LocalMap(untracked_threshold);

    pose.setIdentity();

    prevNbMatch = 0;

//    lastOk = false;

//    lastPosition.setZero();
//    linearVelocity.setZero();
//    lastQuaternion = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
//    angularVelocity = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
}

SparseVO::~SparseVO()
{
    delete orb;
    delete localMap;
}

void SparseVO::extractFeatures()
{
    keypoints.clear();
    descriptors.release();

    orb->operator()(gray, cv::noArray(), keypoints, descriptors);
}

void  SparseVO::setFrame(const cv::Mat& im_rgb, const cv::Mat& im_depth, const cv::Mat& im_gray)
{
    rgb = im_rgb.clone();
    depth = im_depth.clone();
    gray = im_gray.clone();
}

void SparseVO::matchFeatures()
{
    matchedMapPositions.clear();
    matchedKeypointsPositions.clear();
    keypointsMatchesIdx.clear();

    if(localMap->size() > 0)
    {
        localMap->findMatches(keypoints,
                              descriptors,
                              matchedMapPositions,
                              matchedKeypointsPositions,
                              keypointsMatchesIdx,
                              pose.inverse(),
                              cam,
                              rangeMin,
                              rangeMax,
                              rgb);
    }
    else
        keypointsMatchesIdx.resize(keypoints.size(), -1);
}

bool SparseVO::track()
{
    bool valid = false;

//    Eigen::Vector3f new_lin_velocity = pose.translation() - lastPosition;
//    new_lin_velocity = (new_lin_velocity + linearVelocity) * 0.5f;

//    Eigen::Quaternionf quat(pose.linear());
//    Eigen::Quaternionf ang_vel_diff = quat * lastQuaternion.inverse();
//    Eigen::Quaternionf new_ang_vel = ang_vel_diff.slerp(0.5f, angularVelocity);
//    new_ang_vel.normalize();

//    lastQuaternion = quat;
//    angularVelocity = new_ang_vel;
//    lastPosition = pose.translation();
//    linearVelocity = new_lin_velocity;

    if(matchedMapPositions.size() >= 20)
    {
        valid = solver.computePose(pose,
                                   matchedMapPositions,
                                   matchedKeypointsPositions,
                                   cam.fx,
                                   cam.fy,
                                   cam.cx,
                                   cam.cy);
    }

//    if(lastOk && !valid)
//    {
//        pose.translation() += linearVelocity;
//        Eigen::Quaternionf updated_quat = quat * new_ang_vel;
//        updated_quat.normalize();
//        pose.linear() = updated_quat.toRotationMatrix();
//    }

    if(valid)
    {
        //lastOk = true;
        std::cout<<"Sparse VO success"<<std::endl;
    }
    //else
    //    lastOk = false;

    return valid;
}

void SparseVO::updateLocalMap()
{
    //if(localMap->size() < 250 || float(matchedMapPositions.size()) < 0.5f * prevNbMatch || matchedMapPositions.size() < 50)
        localMap->update(pose,
                         cam,
                         rangeMin,
                         rangeMax,
                         depth,
                         keypoints,
                         descriptors,
                         keypointsMatchesIdx);
    localMap->clean();

    prevNbMatch = matchedMapPositions.size();
}

void SparseVO::updateLocalMapMOD(const cv::Mat& index_mat,
                                 const std::vector<bool>& is_static/*,
                                 const cv::Mat& mask*/)
{
    localMap->updateMOD(pose,
                        cam,
                        rangeMin,
                        rangeMax,
                        depth,
                        keypoints,
                        descriptors,
                        keypointsMatchesIdx,
                        index_mat,
                        is_static/*,
                        mask*/);
    prevNbMatch = matchedMapPositions.size();
}

void SparseVO::computeFilteredKeypoints3D(std::vector<cv::KeyPoint>& filtered_keypoints,
                                          cv::Mat& filtered_descriptors,
                                          std::vector<cv::Point3f>& points3D,
                                          float range_min,
                                          float range_max)
{
    filtered_keypoints.clear();
    filtered_descriptors.release();
    points3D.clear();

    for(size_t i = 0; i < keypoints.size(); i++)
    {
        float z = depth.at<float>(keypoints[i].pt.y, keypoints[i].pt.x);

        if(z >= range_min && z <= range_max)
        {
            points3D.push_back(cv::Point3f(z * (keypoints[i].pt.x - cam.cx) / cam.fx,
                                           z * (keypoints[i].pt.y - cam.cy) / cam.fy,
                                           z));
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row(i).clone());
        }
    }
}

void SparseVO::reset(const Eigen::Isometry3f& reset_pose)
{
    pose = reset_pose;
    localMap->reset(pose,
                    cam,
                    rangeMin,
                    rangeMax,
                    depth,
                    gray,
                    keypoints,
                    descriptors);
}

} // supersurfel_fusion
