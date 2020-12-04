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


#ifndef LOCAL_MAP_HPP
#define LOCAL_MAP_HPP

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <supersurfel_fusion/cam_param.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <Eigen/StdVector>


namespace supersurfel_fusion
{

class LocalMap
{

public:
    LocalMap();
    LocalMap(int untracked_thresh);
    void update(const Eigen::Isometry3f cam_to_map,
                const CamParam& cam_param,
                float range_min,
                float range_max,
                const cv::Mat& depth,
                const std::vector<cv::KeyPoint>& frame_keypoints,
                const cv::Mat& frame_descriptors,
                const std::vector<int>& matches_idx);
    void updateMOD(const Eigen::Isometry3f cam_to_map,
                   const CamParam& cam_param,
                   float range_min,
                   float range_max,
                   const cv::Mat& depth,
                   const std::vector<cv::KeyPoint>& frame_keypoints,
                   const cv::Mat& frame_descriptors,
                   const std::vector<int>& matches_idx,
                   const cv::Mat& index_mat,
                   const std::vector<bool>& is_static/*,
                   const cv::Mat& mask*/);
    void clean();
    void findMatches(const std::vector<cv::KeyPoint>& frame_keypoints,
                     const cv::Mat& frame_descriptors,
                     std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& matched_map_positions,
                     std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& matched_frame_positions,
                     std::vector<int>& keypoints_matches_idx,
                     const Eigen::Isometry3f map_to_cam,
                     const CamParam& cam_param,
                     float range_min,
                     float range_max,
                     const cv::Mat& rgb);
    void reset(const Eigen::Isometry3f cam_to_map,
               const CamParam& cam_param,
               float range_min,
               float range_max,
               const cv::Mat& depth,
               const cv::Mat& gray,
               const std::vector<cv::KeyPoint>& frame_keypoints,
               const cv::Mat& frame_descriptors);

    inline const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& getPositions() const {return positions;}
    inline const cv::Mat& getDescriptors() const {return descriptors;}
    inline const std::vector<int>& getCounters() const {return counters;}
    inline std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& setPositions() {return positions;}
    inline cv::Mat& setDescriptors() {return descriptors;}
    inline std::vector<int>& setCounters() {return counters;}
    inline int size() const {return positions.size();}

private:
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> positions;
    cv::Mat descriptors;
    std::vector<int> counters;
    std::vector<int> states;
    int untrackedThresh;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

}; // class LocalMap

} // supersurfel_fusion

#endif // LOCAL_MAP_HPP
