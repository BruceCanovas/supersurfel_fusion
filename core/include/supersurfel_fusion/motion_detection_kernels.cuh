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


#ifndef MOTION_DETECTION_KERNELS_CUH
#define MOTION_DETECTION_KERNELS_CUH

#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <supersurfel_fusion/vector_math.cuh>


namespace supersurfel_fusion
{

__global__ void initSuperpixelsUVFlow(cudaTextureObject_t tex_index,
                                      cudaTextureObject_t tex_flow,
                                      float2* superpixels_uvflow,
                                      int rows,
                                      int cols,
                                      int border);

__global__ void computeSuperpixelsUVFlow(const SuperpixelRGBD* superpixels,
                                         float2* superpixels_uvflow,
                                         int nb_superpixels);

__global__ void detectDynamicSuperpixels(const SuperpixelRGBD* superpixels,
                                         bool* is_static,
                                         float* confidences,
                                         float2* superpixels_uvflow,
                                         cudaTextureObject_t tex_depth_estimate,
                                         float thresh_flow,
                                         int nb_superpixels);

__global__ void computeStaticDynamicImage(cudaTextureObject_t tex_index,
                                          bool* labels,
                                          uchar* mask,
                                          int rows,
                                          int cols,
                                          int step);

__global__ void computeSuperpixelsAdjacency(cudaTextureObject_t texIndex,
                                            cv::cuda::PtrStep<uchar> adjacency_mat,
                                            int width,
                                            int height);

__global__ void computeLabelsImage(cudaTextureObject_t tex_index,
                                   int* labels,
                                   float3* im_label,
                                   int3* colors,
                                   int rows,
                                   int cols,
                                   int step);

} // supersurfel_fusion

#endif // MOTION_DETECTION_KERNELS_CUH
