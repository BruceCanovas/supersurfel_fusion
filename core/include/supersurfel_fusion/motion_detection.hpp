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


#ifndef MOTION_DETECTION_HPP
#define MOTION_DETECTION_HPP

#include <opencv2/optflow.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <supersurfel_fusion/matrix_types.h>
#include "yolo_v2_class.hpp"


namespace supersurfel_fusion
{

class MotionDetection
{

public:

    MotionDetection();
    void computeGeometricClusters(const thrust::host_vector<float3>& positions,
                                  const thrust::host_vector<Mat33>& orientations,
                                  const thrust::host_vector<float>& confidences,
                                  std::vector<std::vector<int>>& clusters,
                                  std::vector<std::vector<int>>& adjacency_list,
                                  std::vector<std::vector<int>>& adjacency_list_all);
    void detectMotion(const cv::Mat& rgb,
                      const cv::Mat& gray,
                      const cv::Mat& depth,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::Mat& descriptors,
                      const thrust::device_vector<float3>& supersurfels_positions,
                      const thrust::device_vector<Mat33>& supersurfels_orientations,
                      thrust::device_vector<float>& supersurfels_confidences,
                      const thrust::device_vector<SuperpixelRGBD>& superpixels,
                      const cv::cuda::GpuMat& index_mat);
    void detectMotionSimple(const cv::Mat& gray,
                            const cv::Mat& depth,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& descriptors,
                            const thrust::device_vector<float3>& supersurfels_positions,
                            const thrust::device_vector<Mat33>& supersurfels_orientations,
                            thrust::device_vector<float>& supersurfels_confidences,
                            const thrust::device_vector<SuperpixelRGBD>& superpixels,
                            const cv::cuda::GpuMat& index_mat);
    void detectMotionCombined(const cv::Mat& rgb,
                              const cv::Mat& gray,
                              const cv::Mat& depth,
                              std::vector<cv::KeyPoint>& keypoints,
                              cv::Mat& descriptors,
                              const thrust::device_vector<float3>& supersurfels_positions,
                              const thrust::device_vector<Mat33>& supersurfels_orientations,
                              thrust::device_vector<float>& supersurfels_confidences,
                              const thrust::device_vector<SuperpixelRGBD>& superpixels,
                              const cv::cuda::GpuMat& index_mat);
    void detectMotionYoloOnly(const cv::Mat& rgb,
                              std::vector<cv::KeyPoint>& keypoints,
                              cv::Mat& descriptors,
                              const thrust::device_vector<float3>& supersurfels_positions,
                              thrust::device_vector<float>& supersurfels_confidences,
                              const thrust::device_vector<SuperpixelRGBD>& superpixels,
                              const cv::cuda::GpuMat& index_mat);
    void setYoloDetector(const std::string& yolo_config_file,
                         const std::string& yolo_weights_file,
                         const std::string& yolo_names_file);
    void detectYolo(const cv::Mat& rgb);
    inline cv::Mat const& getMask() const {return mask;}
    inline std::vector<bool> const& getIsStatic() const {return isStatic;}


private:
    cv::Ptr<cv::DenseOpticalFlow> dof;
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
    cv::Mat prevDescriptors;
    cv::Mat prevGray, prevDepth;
    cv::cuda::GpuMat indexMat, flowMat, depthEstimateMat;
    cv::Ptr<Texture<int>> texIndex;
    cv::Ptr<Texture<float2>> texFlow;
    cv::Ptr<Texture<float>> texDepthEstimate;

    //std::vector<std::vector<cv::KeyPoint>> prevKeypointsVec;
    //std::vector<cv::Mat> prevDescriptorsVec;
    //std::vector<cv::Mat> prevGrayVec, prevDepthVec;

    std::vector<bool> isStatic;

    cv::Mat mask;

    cv::Ptr<Detector> yoloDetector;
    std::vector<std::string> objectsNames;
    std::vector<bbox_t> objects;

    bool useYolo;

}; // class MotionDetection

} // supersurfel_fusion

#endif // MOTION_DETECTION_HPP
