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


#ifndef SPARSE_VO_HPP
#define SPARSE_VO_HPP

#include "ORBextractor.h"
#include <supersurfel_fusion/local_map.hpp>
#include <Eigen/StdVector>
#include <supersurfel_fusion/pnp_solver.hpp>
//#include "KDTreeVectorOfVectorsAdaptor.h"


namespace supersurfel_fusion
{

class SparseVO
{

public:
    SparseVO();
    SparseVO(int nb_features,
             float scale_factor,
             int nb_levels,
             int ini_th_fast,
             int min_th_fast,
             const CamParam& cam_param,
             float range_min,
             float range_max,
             int untracked_threshold);
    ~SparseVO();
    void setFrame(const cv::Mat& im_rgb, const cv::Mat& im_depth, const cv::Mat& im_gray);
    void extractFeatures();
    void matchFeatures();
    bool track();
    void updateLocalMap();
    void updateLocalMapMOD(const cv::Mat& index_mat,
                           const std::vector<bool>& is_static/*,
                           const cv::Mat& mask*/);
    void computeFilteredKeypoints3D(std::vector<cv::KeyPoint>& filtered_keypoints,
                                    cv::Mat& filtered_descriptors,
                                    std::vector<cv::Point3f>& points3D,
                                    float range_min,
                                    float range_max);
    void reset(const Eigen::Isometry3f& reset_pose);
    inline const std::vector<cv::KeyPoint>& getKeypoints() const {return keypoints;}
    inline const cv::Mat& getDescriptors() const {return descriptors;}
    inline std::vector<cv::KeyPoint>& setKeypoints() {return keypoints;}
    inline cv::Mat& setDescriptors() {return descriptors;}
    inline const Eigen::Isometry3f& getPose() const {return pose;}
    inline Eigen::Isometry3f& setPose() {return pose;}
    inline const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& getLocalMapPoints() const {return localMap->getPositions();}
    inline const std::vector<int>& getInlierMarks() const {return solver.getInlierMarks();}
    inline int getNbInliers() const {return solver.getNbInliers();}
    inline const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& getMatchedMapPositions() const {return matchedMapPositions;}
    inline const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& getMatchedKeypointsPositions() const {return matchedKeypointsPositions;}

private:
    cv::Mat rgb, depth, gray;
    ORB_SLAM2::ORBextractor *orb;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> matchedMapPositions;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> matchedKeypointsPositions;
    std::vector<int> keypointsMatchesIdx;
    CamParam cam;
    float rangeMin, rangeMax;
    LocalMap* localMap;
    Eigen::Isometry3f pose;
    //Eigen::Vector3f lastPosition, linearVelocity;
    //Eigen::Quaternionf lastQuaternion, angularVelocity;
    PnPSolver solver;
    int prevNbMatch;
    //bool lastOk;

}; // class SparseVO

} // supersurfel_fusion

#endif //SPARSE_VO_HPP
