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


#include <supersurfel_fusion/motion_detection_kernels.cuh>
#include <stdio.h>


namespace supersurfel_fusion
{

__global__ void initSuperpixelsUVFlow(cudaTextureObject_t tex_index,
                                      cudaTextureObject_t tex_flow,
                                      float2* superpixels_uvflow,
                                      int rows,
                                      int cols,
                                      int border)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if(x < border || x >= cols - border || y < border || y >= rows - border)
        return;

    int index = tex2D<int>(tex_index, x, y);

    float2 uv = tex2D<float2>(tex_flow, x, y);

    if(isfinite(uv.x) && isfinite(uv.y) && length(uv) > 2.0f && length(uv) <= 50.0f)
    {
        atomicAdd(&(superpixels_uvflow[index].x), uv.x);
        atomicAdd(&(superpixels_uvflow[index].y), uv.y);
    }
}

__global__ void computeSuperpixelsUVFlow(const SuperpixelRGBD* superpixels,
                                         float2* superpixels_uvflow,
                                         int nb_superpixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= nb_superpixels)
        return;

    float2 uv = superpixels_uvflow[index] / superpixels[index].size.x;
    superpixels_uvflow[index] = uv;
}

__global__ void detectDynamicSuperpixels(const SuperpixelRGBD* superpixels,
                                         bool* is_static,
                                         float* confidences,
                                         float2* superpixels_uvflow,
                                         cudaTextureObject_t tex_depth_estimate,
                                         float thresh_flow,
                                         int nb_superpixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= nb_superpixels)
        return;

    if(!is_static[index])
    {
        confidences[index] = -1.0f;
        return;
    }

    float2 uv = superpixels_uvflow[index] / superpixels[index].size.x;
    superpixels_uvflow[index] = uv;

    int row = lroundf(superpixels[index].xy_rg.y);
    int col = lroundf(superpixels[index].xy_rg.x);

    float4 theta_b = superpixels[index].theta_b;
    float z = 1.0f / (theta_b.x * superpixels[index].xy_rg.x + theta_b.y * superpixels[index].xy_rg.y + theta_b.z);
    float z_estimate = tex2D<float>(tex_depth_estimate, col, row);

    float z_diff = 0.0f;
    float thresh_depth = 1000.0f;

    if(isfinite(z) && isfinite(z_estimate) && z > 0.1f && z < 5.0f && z_estimate > 0.1f && z_estimate < 5.0f)
    {
        thresh_depth = 20.0f * (0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f));
        z_diff = fabsf(z_estimate - z);
    }

    if(length(uv) > thresh_flow/* || z_diff > thresh_depth*/)
    {
        is_static[index] = false;
        confidences[index] = -1.0f;
    }
}

__global__ void computeStaticDynamicImage(cudaTextureObject_t tex_index,
                                          bool* labels,
                                          uchar* mask,
                                          int rows,
                                          int cols,
                                          int step)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if(x >= cols || y >= rows)
        return;

    int tid = y*step + x;

    int index = tex2D<int>(tex_index, x, y);

    uchar val = 0;

    if(labels[index] == false)
        val = 255;

    mask[tid] = val;
}

__global__ void computeSuperpixelsAdjacency(cudaTextureObject_t texIndex,
                                            cv::cuda::PtrStep<uchar> adjacency_mat,
                                            int width,
                                            int height)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=width-1 || y>=height-1 || x==0 || y==0)
        return;

    int k = tex2D<int>(texIndex, x, y);
    int k_left = tex2D<int>(texIndex, x-1, y);
    int k_right = tex2D<int>(texIndex, x+1, y);
    int k_up = tex2D<int>(texIndex, x, y+1);
    int k_down = tex2D<int>(texIndex, x, y-1);
//    int k_up_right = tex2D<int>(texIndex, x+1, y+1);
//    int k_up_left = tex2D<int>(texIndex, x-1, y+1);
//    int k_down_right = tex2D<int>(texIndex, x+1, y-1);
//    int k_down_left = tex2D<int>(texIndex, x-1, y-1);

    if(k != k_left)
    {
        adjacency_mat(k, k_left) = 1;
        adjacency_mat(k_left, k) = 1;
    }

    if(k != k_right)
    {
        adjacency_mat(k, k_right) = 1;
        adjacency_mat(k_right, k) = 1;
    }

    if(k != k_up)
    {
        adjacency_mat(k, k_up) = 1;
        adjacency_mat(k_up, k) = 1;
    }

    if(k != k_down)
    {
        adjacency_mat(k, k_down) = 1;
        adjacency_mat(k_down, k) = 1;
    }

//    if(k != k_up_right)
//    {
//        adjacency_mat(k, k_up_right) = 1;
//        adjacency_mat(k_up_right, k) = 1;
//    }

//    if(k != k_up_left)
//    {
//        adjacency_mat(k, k_up_left) = 1;
//        adjacency_mat(k_up_left, k) = 1;
//    }

//    if(k != k_down_right)
//    {
//        adjacency_mat(k, k_down_right) = 1;
//        adjacency_mat(k_down_right, k) = 1;
//    }

//    if(k != k_down_left)
//    {
//        adjacency_mat(k, k_down_left) = 1;
//        adjacency_mat(k_down_left, k) = 1;
//    }
}

__global__ void computeLabelsImage(cudaTextureObject_t tex_index,
                                   int* labels,
                                   float3* im_label,
                                   int3* colors,
                                   int rows,
                                   int cols,
                                   int step)
{

    int x = blockIdx.x *blockDim.x + threadIdx.x; // col
    int y = blockIdx.y *blockDim.y + threadIdx.y; // row

    if(x >= cols || y >= rows)
        return;

    unsigned int tid = y*step + x;

    int index = tex2D<int>(tex_index, x, y);

    float3 color = make_float3(0, 0, 0);

    if(labels[index]>=0)
        color = make_float3(float(colors[labels[index]].x), float(colors[labels[index]].y), float(colors[labels[index]].z));

    im_label[tid] = color;
}

} // supersurfel_fusion
