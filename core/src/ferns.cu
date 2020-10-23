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


#include <supersurfel_fusion/ferns.hpp>
#include <supersurfel_fusion/cuda_error_check.h>
#include <supersurfel_fusion/ferns_kernels.cuh>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <opencv2/cudawarping.hpp>


namespace supersurfel_fusion
{

Ferns::Ferns(int n, int pyrLevel, float maxDepth)
    : n(n)
    , pyrLevel(pyrLevel)
    , maxDepth(maxDepth)
    , newFrame(false)
{
}

Ferns::~Ferns()
{
    delete(texColor);
    delete(texDepth);
}

void Ferns::processFerns(const cv::cuda::GpuMat& rgb,
                         const cv::cuda::GpuMat& depth,
                         const float threshold)
{
    if(pyrColor.empty() || rgb.size()!=pyrColor[0].size())
    {
        init(rgb, depth);
    }

    cv::cuda::cvtColor(rgb, pyrColor[0], cv::COLOR_BGR2RGBA);
    depth.copyTo(pyrDepth[0]);

    for(int l=1; l<pyrLevel; l++)
    {
        cv::cuda::resize(pyrColor[l-1], pyrColor[l], cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);
        cv::cuda::resize(pyrDepth[l-1], pyrDepth[l], cv::Size(0,0), 0.5, 0.5, cv::INTER_NEAREST);
    }

    computeCodes_kernel<<<(n+63)/64, 64>>>(thrust::raw_pointer_cast(codesDev.data()),
                                           thrust::raw_pointer_cast(ferns.data()),
                                           texColor->getTextureObject(),
                                           texDepth->getTextureObject(),
                                           n);

    cudaDeviceSynchronize();
    CudaCheckError();

    codesHost = codesDev;

    if(!keyFrames.empty())
    {
        coOccurences.resize(keyFrames.size());
        memset(&coOccurences[0], 0, sizeof(int)*coOccurences.size());

        for(int k = 0; k < n; k++)
        {
            uint8_t code = codesHost[k];
            for(size_t i = 0; i<fernIds[k][code].size(); i++)
            {
                coOccurences[fernIds[k][code][i]]++;
            }
        }
    }

    float minDiss = 1.f;
    int bestKey = 0;
    for(size_t i=0; i<coOccurences.size(); i++)
    {
        float dissim = (float)(n-coOccurences[i])/(float)n;
        if(dissim < minDiss)
        {
            minDiss = dissim;
            bestKey = i;
        }
    }

    std::cout << "bestKey : " << bestKey << std::endl;

    //if(!keyFrames.empty())
    bestKeyFrameId = bestKey;

    if(minDiss>threshold)
    {
        for(int k = 0; k < n; k++)
        {
            uint8_t code = codesHost[k];
            fernIds[k][code].push_back(keyFrames.size());
        }

        newFrame = true;

        bestKeyFrameId = keyFrames.size();

        //return true;
        //return;
    }

    //return false;
}

void Ferns::addKeyFrame(const Mat33& R,
                        float3 t,
                        int stamp,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const cv::Mat& descriptors,
                        const std::vector<cv::Point3f>& keypoints3D,
                        const thrust::host_vector<float3>& positions,
                        const thrust::host_vector<float3>& colors,
                        const thrust::host_vector<Mat33>& orientations,
                        const thrust::host_vector<float>& confidences)
{
    KeyFrame kf;
    kf.fernId = keyFrames.size();
    kf.keypoints = keypoints;
    kf.descriptors = descriptors.clone();
    kf.keypoints3D = keypoints3D;

    for(size_t i = 0; i < confidences.size(); i++)
    {
        if(confidences[i] > 0.0f)
        {
            kf.positions.push_back(positions[i]);
            kf.colors.push_back(colors[i]);
            kf.orientations.push_back(orientations[i]);
        }
    }

    keyFrames.push_back(kf);

    StampedPose stamped_pose;
    stamped_pose.R = R;
    stamped_pose.t = t;
    stamped_pose.stamp = stamp;

    poseGraph.push_back(stamped_pose);

    newFrame = false;
}

void Ferns::init(const cv::cuda::GpuMat& rgb,
                 const cv::cuda::GpuMat& depth)
{
    pyrColor.resize(pyrLevel);
    pyrDepth.resize(pyrLevel);
    ferns.resize(n);
    fernIds.resize(n);
    codesDev.resize(n);
    codesHost.resize(n);
    keyFrames.clear();

    cv::cuda::cvtColor(rgb, pyrColor[0], cv::COLOR_BGR2RGBA);
    depth.copyTo(pyrDepth[0]);

    for(int l=1; l<pyrLevel; l++)
    {
        cv::cuda::resize(pyrColor[l-1], pyrColor[l], cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);
        cv::cuda::resize(pyrDepth[l-1], pyrDepth[l], cv::Size(0,0), 0.5, 0.5, cv::INTER_NEAREST);
    }

    texColor = new Texture<uchar4>(pyrColor[pyrLevel-1]);
    texDepth = new Texture<float>(pyrDepth[pyrLevel-1]);

    initFerns_kernel<<<(n+63)/64, 64>>>(thrust::raw_pointer_cast(ferns.data()),
                                        maxDepth,
                                        pyrColor[pyrLevel-1].cols,
                                        pyrColor[pyrLevel-1].rows,
                                        n);
    cudaDeviceSynchronize();
    CudaCheckError();

}

} // supersurfel_fusion
