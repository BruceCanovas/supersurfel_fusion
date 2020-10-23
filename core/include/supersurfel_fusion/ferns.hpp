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


#ifndef FERNS_HPP
#define FERNS_HPP

#include <supersurfel_fusion/ferns_types.hpp>
#include <array>
#include <thrust/device_vector.h>
#include <supersurfel_fusion/texture.hpp>


namespace supersurfel_fusion
{

class Ferns
{

public:
    Ferns(int n, int pyrlevel, float maxDepth);
    ~Ferns();
    void processFerns(const cv::cuda::GpuMat& rgb,
                      const cv::cuda::GpuMat& depth,
                      const float threshold);
    void addKeyFrame(const Mat33& R,
                     float3 t,
                     int stamp,
                     const std::vector<cv::KeyPoint>& keypoints,
                     const cv::Mat& descriptors,
                     const std::vector<cv::Point3f>& keypoints3D,
                     const thrust::host_vector<float3>& positions,
                     const thrust::host_vector<float3>& colors,
                     const thrust::host_vector<Mat33>& orientations,
                     const thrust::host_vector<float>& confidences);


    void init(const cv::cuda::GpuMat& rgb,
              const cv::cuda::GpuMat& depth);
    inline const int& getBestKeyFrameId() const {return bestKeyFrameId;}
    inline const KeyFrame& getBestMatch() const {return keyFrames[bestKeyFrameId];}
    inline const StampedPose& getBestPose() const {return poseGraph[bestKeyFrameId];}
    inline bool isNewFrame() const {return newFrame;}
    inline const std::vector<StampedPose>& getPoseGraph() const {return poseGraph;}
    inline std::vector<StampedPose>& setPoseGraph() {return poseGraph;}

private:
    thrust::device_vector<Fern> ferns;
    thrust::device_vector<uint8_t> codesDev;
    thrust::host_vector<uint8_t> codesHost;

    typedef std::array<std::vector<int>, 16> fernIds_t;
    std::vector<fernIds_t> fernIds;
    std::vector<int> coOccurences;

    std::vector<KeyFrame> keyFrames;
    std::vector<StampedPose> poseGraph;

    int n;
    int pyrLevel;
    float threshold;
    float maxDepth;

    std::vector<cv::cuda::GpuMat> pyrColor, pyrDepth;
    Texture<uchar4> *texColor;
    Texture<float> *texDepth;


    bool newFrame;

    int bestKeyFrameId;

}; // class Ferns

} // supersurfel_fusion

#endif // FERNS_HPP
