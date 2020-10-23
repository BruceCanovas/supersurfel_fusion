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


#ifndef TPS_RGBD_HPP
#define TPS_RGBD_HPP

#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand_kernel.h>
#include <supersurfel_fusion/texture.hpp>


namespace supersurfel_fusion
{

struct SuperpixelRGBD
{
    float4 xy_rg;   // image position, red and green
    float4 theta_b; // slanted plane and blue
    float4 size;    // pixel number
};

struct SuperpixelRGBDCoeffs
{
    float x, y, r, g, b, n; // rgb coeff
    float dx, dy, dxx, dyy, dxy, dxd, dyd, dd, dn; // disp coeff
};

struct FilterDataRGBD
{
    float3 X;
    float3 Z;
    float px, py;
};

class TPS_RGBD
{

public:
    TPS_RGBD();

    TPS_RGBD(int cell_size,
             float lambda_pos,
             float lambda_bound,
             float lambda_size,
             float lambda_disp,
             float thresh_disp,
             int nb_iters,
             bool use_ransac,
             int nb_samples,
             int filter_iter,
             float filter_alpha,
             float filter_beta,
             float filter_thresh);

    void compute(const cv::cuda::GpuMat& rgb,
                 const cv::cuda::GpuMat& depth);

    void filter();

    inline const cv::cuda::GpuMat& getIndexImage() const {return indexMat;}
    inline const cv::cuda::GpuMat& getBoundaryImage() const {return boundaryMat;}
    inline const cv::cuda::GpuMat& getDispImage() const {return dispMat;}
    inline const cv::cuda::GpuMat& getInliersImage() const {return inliersMat;}
    inline const cv::cuda::GpuMat& getRGBAImage() const {return imgRGBA;}

    inline int getNbSuperpixels() const {return nbSuperpixels;}
    inline const thrust::device_vector<SuperpixelRGBD>& getSuperpixels() const {return superpixels;}
    inline const cv::Ptr<Texture<uchar4>>& getTexRGBA() const {return texRGBA;}
    inline const cv::Ptr<Texture<float>>& getTexDisp() const {return texDisp;}
    inline const cv::Ptr<Texture<int>>& getTexIndex() const {return texIndex;}
    inline const cv::Ptr<Texture<int>>& getTexBound() const {return texBound;}
    inline const cv::Ptr<Texture<unsigned char>>& getTexInliers() const {return texInliers;}

    void computePreviewImage(cv::cuda::GpuMat& previewImg);
    void computeDepthImage(cv::cuda::GpuMat& depthMat);

private:
    cv::cuda::GpuMat imgRGBA, indexMat, boundaryMat, segmentedMat, dispMat, inliersMat;

    cv::cuda::GpuMat displayImgGpu;

    thrust::device_vector<SuperpixelRGBD> superpixels;
    thrust::device_vector<SuperpixelRGBDCoeffs> coeffs;

    thrust::device_vector<FilterDataRGBD> filterData;

    thrust::device_vector<float4> samples;
    thrust::device_vector<curandState> randStates;

    cv::Mat displayImg, displayImg2;

    cv::Ptr<Texture<uchar4>> texRGBA;
    cv::Ptr<Texture<float>> texDisp;
    cv::Ptr<Texture<int>> texIndex, texBound;
    cv::Ptr<Texture<unsigned char>> texInliers;

    int cellSize, gridSizeX, gridSizeY;

    float lambdaPos, lambdaBound, lambdaSize, lambdaDisp, threshDisp;
    int nbIters;
    bool useRansac;

    cv::Rect roi;

    int nbSuperpixels;
    int nbSamples;

    int filterIter;
    float filterAlpha, filterBeta, filterThresh;

    void initMat(const cv::cuda::GpuMat& img);

}; // class TPS_RGBD

} // namespace supersurfel_fusion

#endif // TPS_RGBD_HPP
