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


#include <supersurfel_fusion/local_map.hpp>
#include "gms_matcher.h"


namespace supersurfel_fusion
{

LocalMap::LocalMap()
    : untrackedThresh(10)
{
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

}

LocalMap::LocalMap(int untracked_thresh)
    : untrackedThresh(untracked_thresh)
{
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
}

void LocalMap::update(const Eigen::Isometry3f cam_to_map,
                      const CamParam& cam_param,
                      float range_min,
                      float range_max,
                      const cv::Mat& depth,
                      const std::vector<cv::KeyPoint>& frame_keypoints,
                      const cv::Mat& frame_descriptors,
                      const std::vector<int>& matches_idx)
{
    for(size_t i = 0; i < matches_idx.size(); i++)
    {
        float z = depth.at<float>(frame_keypoints[i].pt.y, frame_keypoints[i].pt.x);

        if(z >= range_min && z <= range_max)
        {
            Eigen::Vector3f p(z * (frame_keypoints[i].pt.x - cam_param.cx) / cam_param.fx,
                              z * (frame_keypoints[i].pt.y - cam_param.cy) / cam_param.fy,
                              z);
            p = cam_to_map * p;

            if(matches_idx[i] > 0) // replace map point with new match
            {
                positions[matches_idx[i]] = p;
                frame_descriptors.row(i).copyTo(descriptors.row(matches_idx[i]));
            }
            else // insert new point
            {
                positions.push_back(p);
                counters.push_back(0);
                descriptors.push_back(frame_descriptors.row(i).clone());
            }
        } 
    }
}

void LocalMap::updateMOD(const Eigen::Isometry3f cam_to_map,
                         const CamParam& cam_param,
                         float range_min,
                         float range_max,
                         const cv::Mat& depth,
                         const std::vector<cv::KeyPoint>& frame_keypoints,
                         const cv::Mat& frame_descriptors,
                         const std::vector<int>& matches_idx,
                         const cv::Mat& index_mat,
                         const std::vector<bool>& is_static/*,
                         const cv::Mat& mask*/)
{
    for(size_t i = 0; i < matches_idx.size(); i++)
    {
        float z = depth.at<float>(frame_keypoints[i].pt.y, frame_keypoints[i].pt.x);

        if(z >= range_min && z <= range_max && is_static[index_mat.at<int>(frame_keypoints[i].pt.y, frame_keypoints[i].pt.x)])
        {
            Eigen::Vector3f p(z * (frame_keypoints[i].pt.x - cam_param.cx) / cam_param.fx,
                              z * (frame_keypoints[i].pt.y - cam_param.cy) / cam_param.fy,
                              z);
            p = cam_to_map * p;

            if(matches_idx[i] > 0) // replace map point with new match
            {
                positions[matches_idx[i]] = p;
                frame_descriptors.row(i).copyTo(descriptors.row(matches_idx[i]));
            }
            else // insert new point
            {
                positions.push_back(p);
                counters.push_back(0);
                descriptors.push_back(frame_descriptors.row(i).clone());
            }
        }
    }

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> cleaned_positions;
    std::vector<int> cleaned_counters;
    cv::Mat cleaned_descriptors;

    Eigen::Isometry3f map_to_cam = cam_to_map.inverse();

    for(size_t i = 0; i < positions.size(); i++)
    {
        Eigen::Vector3f p = map_to_cam * positions[i];

        float col = cam_param.fx * p(0) / p(2) + cam_param.cx;
        float row = cam_param.fy * p(1) / p(2) + cam_param.cy;

        if(col >= 0 && col < cam_param.width && row >= 0 && row < cam_param.height)
        {
            if(!is_static[index_mat.at<int>(row, col)])
            //if(mask.at<uchar>(row, col))
            {
                //std::cout<<"remove dynamic map point"<<std::endl;
                continue;
            }
        }

        if(counters[i] < untrackedThresh)
        {
            cleaned_counters.push_back(counters[i]);
            cleaned_positions.push_back(positions[i]);
            cleaned_descriptors.push_back(descriptors.row(i).clone());
        }
    }

    std::swap(cleaned_positions, positions);
    std::swap(cleaned_counters, counters);
    descriptors = cleaned_descriptors.clone();
}

void LocalMap::reset(const Eigen::Isometry3f cam_to_map,
                     const CamParam& cam_param,
                     float range_min,
                     float range_max,
                     const cv::Mat& depth,
                     const cv::Mat& gray,
                     const std::vector<cv::KeyPoint>& frame_keypoints,
                     const cv::Mat& frame_descriptors)
{
    positions.clear();
    counters.clear();
    descriptors.release();

    for(size_t i = 0; i < frame_keypoints.size(); i++)
    {
        float z = depth.at<float>(frame_keypoints[i].pt.y, frame_keypoints[i].pt.x);

        if(z >= range_min && z <= range_max)
        {
            Eigen::Vector3f p(z * (frame_keypoints[i].pt.x - cam_param.cx) / cam_param.fx,
                              z * (frame_keypoints[i].pt.y - cam_param.cy) / cam_param.fy,
                              z);
            p = cam_to_map * p;

            positions.push_back(p);
            counters.push_back(0);
            descriptors.push_back(frame_descriptors.row(i).clone());
        }
    }
}

void LocalMap::clean()
{
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> cleaned_positions;
    std::vector<int> cleaned_counters;
    cv::Mat cleaned_descriptors;

    for(size_t i = 0; i < positions.size(); i++)
    {
        if(counters[i] < untrackedThresh)
        {
            cleaned_counters.push_back(counters[i]);
            cleaned_positions.push_back(positions[i]);
            cleaned_descriptors.push_back(descriptors.row(i).clone());
        }
    }

    std::swap(cleaned_positions, positions);
    std::swap(cleaned_counters, counters);
    descriptors = cleaned_descriptors.clone();
}

void LocalMap::findMatches(const std::vector<cv::KeyPoint>& frame_keypoints,
                           const cv::Mat& frame_descriptors,
                           std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& matched_map_positions,
                           std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& matched_frame_positions,
                           std::vector<int>& keypoints_matches_idx,
                           const Eigen::Isometry3f map_to_cam,
                           const CamParam& cam_param,
                           float range_min,
                           float range_max,
                           const cv::Mat& rgb)
{
    //cv::Mat vis;
    //cv::cvtColor(rgb, vis, CV_RGB2BGR);

    keypoints_matches_idx.resize(frame_keypoints.size(), -1);

    cv::Mat map_descriptors;
    std::vector<cv::KeyPoint> map_keypoints;
    std::vector<int> map_idx;

    for(size_t i = 0; i < positions.size(); i++)
    {
        Eigen::Vector3f p_view = map_to_cam * positions[i];
        float z = p_view(2);

        if(z >= range_min && z <= range_max)
        {
           cv::Point2f proj(cam_param.fx * p_view(0) / z + cam_param.cx,
                            cam_param.fy * p_view(1) / z + cam_param.cy);

            if(proj.x >= 0 && proj.x < cam_param.width && proj.y >= 0 && proj.y < cam_param.height) // && proj not in dynamic part
            {
                cv::KeyPoint kp;
                kp.pt = proj;
                map_keypoints.push_back(kp);
                map_descriptors.push_back(descriptors.row(i).clone());
                map_idx.push_back(i);

                //cv::drawMarker(vis, cv::Point(int(proj.x), int(proj.y)), cv::Scalar(0,0,255), cv::MARKER_CROSS, 10/* marker_size*/, 1/* thickness*/, 8/* line_type*/);
            }
        }

        counters[i]++;
    }

    std::vector<cv::DMatch> matches_bf;
    cv::cuda::GpuMat map_descriptors_d(map_descriptors), frame_descriptors_d(frame_descriptors);
    matcher->match(frame_descriptors_d, map_descriptors_d, matches_bf);

    std::vector<bool> inliers_states;
    cv::Size img_size(cam_param.width, cam_param.height);
    gms_matcher gms(frame_keypoints, img_size, map_keypoints, img_size, matches_bf);
    int nb_inliers = gms.GetInlierMask(inliers_states, false, false);

    for(size_t j = 0; j < inliers_states.size(); j++)
    {
        if(inliers_states[j] && cv::norm(frame_keypoints[matches_bf[j].queryIdx].pt - map_keypoints[matches_bf[j].trainIdx].pt) < 100.0f)
        {
            matched_map_positions.push_back(positions[map_idx[matches_bf[j].trainIdx]]);
            cv::Point2f pt = frame_keypoints[matches_bf[j].queryIdx].pt;
            matched_frame_positions.push_back(Eigen::Vector2f(pt.x, pt.y));
            keypoints_matches_idx[matches_bf[j].queryIdx] = map_idx[matches_bf[j].trainIdx];

            //cv::Point2f proj = map_keypoints[matches_bf[j].trainIdx].pt;
            //cv::drawMarker(vis, cv::Point(int(proj.x), int(proj.y)), cv::Scalar(0,255,0), cv::MARKER_SQUARE, 10/* marker_size*/, 1/* thickness*/, 8/* line_type*/);

            counters[map_idx[matches_bf[j].trainIdx]]--;
        }
    }

    //cv::imshow("Local map features", vis);
    //cv::waitKey(1);
}

} // supersurfel_fusion
