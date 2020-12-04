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


#ifndef SUPERSURFEL_FUSION_KERNELS_CUH
#define SUPERSURFEL_FUSION_KERNELS_CUH

#include <supersurfel_fusion/supersurfel_fusion.hpp>
#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <opencv2/cudev.hpp>
#include <supersurfel_fusion/matrix_math.cuh>
#include <supersurfel_fusion/cuda_utils_dev.cuh>
#include <supersurfel_fusion/reduce_dev.cuh>
#include <stdio.h>

#define PI 3.141592654f


namespace supersurfel_fusion
{

__global__ void computeSupersurfelCoeffs(float3* positions,
                                         Cov3* shapes,
                                         float3* colors,
                                         float* confidences,
                                         cudaTextureObject_t texImg,
                                         cudaTextureObject_t texDepth,
                                         cudaTextureObject_t texIndex,
                                         cudaTextureObject_t texInliers,
                                         cudaTextureObject_t texBound,
                                         int width,
                                         int height,
                                         float fx,
                                         float fy,
                                         float cx,
                                         float cy);

__global__ void computeSupersurfels(float3* positions,
                                    float3* colors,
                                    int2* stamps,
                                    Mat33* orientations,
                                    Cov3* shapes,
                                    float2* dims,
                                    float* confidences,
                                    float z_min,
                                    float z_max,
                                    int stamp,
                                    int nb_supersurfels);

__global__ void filterModel(float3* positions,
                            int2* stamps,
                            float* confidences,
                            int* states,
                            int current_stamp,
                            int delta_t,
                            float conf_thresh,
                            cudaTextureObject_t texDepth,
                            int* nb_removed,
                            int* nb_active,
                            Mat33 R,
                            float3 t,
                            float fx,
                            float fy,
                            float cx,
                            float cy,
                            float z_min,
                            float z_max,
                            int width,
                            int height,
                            int nb_supersurfels);

__global__ void applyTransformSuperSurfel(float3* positions,
                                          Mat33* orientations,
                                          Cov3* shapes,
                                          float* confidences,
                                          Mat33 R,
                                          float3 t,
                                          int nb_supersurfels);

__global__ void fuseSupersurfels(const float3* frame_positions,
                                 const float3* frame_colors,
                                 const Mat33* frame_orientations,
                                 const Cov3* frame_shapes,
                                 const float2* frame_dims,
                                 const float* frame_confidences,
                                 float3* model_positions,
                                 float3* model_colors,
                                 int2* model_stamps,
                                 Mat33* model_orientations,
                                 Cov3* model_shapes,
                                 float2* model_dims,
                                 float* model_confidences,
                                 bool* matched,
                                 cudaTextureObject_t tex_index,
                                 Mat33 R,
                                 float3 t,
                                 float fx,
                                 float fy,
                                 float cx,
                                 float cy,
                                 float zmin,
                                 float zmax,
                                 int width,
                                 int height,
                                 int stamp,
                                 int* nb_match,
                                 int model_length);

__global__ void insertSupersurfels(const float3* frame_positions,
                                   const float3* frame_colors,
                                   const Mat33* frame_orientations,
                                   const Cov3* frame_shapes,
                                   const float2* frame_dims,
                                   const float* frame_confidences,
                                   float3* model_positions,
                                   float3* model_colors,
                                   int2* model_stamps,
                                   Mat33* model_orientations,
                                   Cov3* model_shapes,
                                   float2* model_dims,
                                   float* model_confidences,
                                   Mat33 R,
                                   float3 t,
                                   int stamp,
                                   bool* matched,
                                   int* model_length,
                                   int frame_length,
                                   int max_model_length);

__global__ void extractLocalPointCloudKernel(const float3* model_positions,
                                             const Mat33* model_orientations,
                                             const float* model_confidences,
                                             float3* cloud_positions,
                                             float3* cloud_normals,
                                             float conf_thresh,
                                             Mat33 R,
                                             float3 t,
                                             float radius,
                                             int* nb_points,
                                             int nb_supersurfels);

__global__ void findBestMatches(const float3* frame_positions,
                                const float3* frame_colors,
                                const Mat33* frame_orientations,
                                //const Cov3* frame_shapes,
                                //const float2* frame_dims,
                                const float* frame_confidences,
                                float3* model_positions,
                                float3* model_colors,
                                //int2* model_stamps,
                                Mat33* model_orientations,
                                //Cov3* model_shapes,
                                //float2* model_dims,
                                float* model_confidences,
                                bool* matched,
                                float2* idx_scores,
                                cudaTextureObject_t tex_index,
                                Mat33 R,
                                float3 t,
                                float fx,
                                float fy,
                                float cx,
                                float cy,
                                float z_min,
                                float z_max,
                                int width,
                                int height,
                                //int stamp,
                                //int* nb_match,
                                int model_length);

__global__ void updateSupersurfels(const float3* frame_positions,
                                   const float3* frame_colors,
                                   const Cov3* frame_shapes,
                                   const float* frame_confidences,
                                   float3* model_positions,
                                   float3* model_colors,
                                   int2* model_stamps,
                                   Mat33* model_orientations,
                                   Cov3* model_shapes,
                                   float2* model_dims,
                                   float* model_confidences,
                                   bool* matched,
                                   float2* idx_scores,
                                   Mat33 R,
                                   float3 t,
                                   int stamp,
                                   int frame_length);

__global__ void findInactives(bool* is_inactive,
                              float3* positions,
                              int2* stamps,
                              float* confidences,
                              int current_stamp,
                              int active_window,
                              float conf_thresh,
                              Mat33 R,
                              float3 t,
                              float fx,
                              float fy,
                              float cx,
                              float cy,
                              float z_min,
                              float z_max,
                              int width,
                              int height,
                              int nb_supersurfels);

} // namespace supersurfel_fusion

#endif // SUPERSURFEL_FUSION_KERNELS_CUH
