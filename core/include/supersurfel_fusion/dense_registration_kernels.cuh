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


#ifndef DENSE_REGISTRATION_KERNELS_CUH
#define DENSE_REGISTRATION_KERNELS_CUH

#include <supersurfel_fusion/dense_registration_types.hpp>
#include <supersurfel_fusion/cuda_utils_dev.cuh>
#include <supersurfel_fusion/reduce_dev.cuh>
#include <supersurfel_fusion/matrix_math.cuh>

#define PI 3.141592654f


namespace supersurfel_fusion
{

__global__ void makeCorrespondences(const float3* source_positions,
                                    const float3* source_colors,
                                    const Mat33* source_orientations,
                                    const float* source_confidences,
                                    const float3* target_positions,
                                    const float3* target_colors,
                                    const Mat33* target_orientations,
                                    const float* target_confidences,
                                    float3* matched_source_normals,
                                    float3* matched_source_positions,
                                    float3* matched_target_normals,
                                    float3* matched_target_positions,
                                    bool* valids,
                                    Mat33 R,
                                    float3 t,
                                    float fx,
                                    float fy,
                                    float cx,
                                    float cy,
                                    cudaTextureObject_t tex_index,
                                    cudaTextureObject_t tex_depth,
                                    int width,
                                    int height,
                                    int nb_supersurfels);

__device__ inline void operator+=(MotionTrackingData& a, const MotionTrackingData& b)
{
    #pragma unroll
    for(int i = 0; i < 21; i++)
        a.JtJ[i] += b.JtJ[i];

    #pragma unroll
    for(int i = 0; i < 6; i++)
        a.Jtr[i] += b.Jtr[i];

    //a.r += b.r;
}

__device__ inline void atomicAdd(MotionTrackingData* a, const MotionTrackingData& b)
{
    #pragma unroll
    for(int i = 0; i < 21; i++)
        ::atomicAdd((float*) &(a->JtJ[i]), b.JtJ[i]);

    #pragma unroll
    for(int i = 0; i < 6; i++)
        ::atomicAdd((float*) &(a->Jtr[i]), b.Jtr[i]);

    //::atomicAdd((float*) &(a->r), b.r);
}

template <int BLOCK_SIZE>
__global__ void buildSymmetricPoint2PlaneSystem(MotionTrackingData* system,
                                                const float3* source_positions,
                                                const float3* source_normals,
                                                const float3* target_positions,
                                                const float3* target_normals,
                                                float3 source_centroid,
                                                float3 target_centroid,
                                                float scale,
                                                int nb_pairs)
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ MotionTrackingData sh_data[BLOCK_SIZE];

    #pragma unroll
    for(int i = 0; i < 21; i++)
        sh_data[tid].JtJ[i] = 0.0f;

    #pragma unroll
    for(int i = 0; i < 6; i++)
        sh_data[tid].Jtr[i] = 0.0f;

    //sh_data[tid].r = 0.0f;

    if(id < nb_pairs)
    {
        float3 ps = scale * (source_positions[id] - source_centroid);
        float3 pt = scale * (target_positions[id] - target_centroid);
        float3 ns = normalize(source_normals[id]);
        float3 nt = normalize(target_normals[id]);

        //float w = dot(ns, nt);
        float w = 1.0f;

        float3 d = pt - ps;
        float3 c1 = cross(pt, ns);
        float3 c2 = cross(ps, nt);
        float dn1 = dot(d, ns);
        float dn2 = dot(d, nt);

        float x1[6] = {c1.x, c1.y, c1.z, ns.x, ns.y, ns.z};
        float x2[6] = {c2.x, c2.y, c2.z, nt.x, nt.y, nt.z};

        for(int i = 0; i < 6; i++)
            sh_data[tid].Jtr[i] = w * (dn1 * x1[i] + dn2 * x2[i]);

        sh_data[tid].JtJ[0] = w * (x1[0] * x1[0] + x2[0] * x2[0]);
        sh_data[tid].JtJ[1] = w * (x1[0] * x1[1] + x2[0] * x2[1]);
        sh_data[tid].JtJ[2] = w * (x1[0] * x1[2] + x2[0] * x2[2]);
        sh_data[tid].JtJ[3] = w * (x1[0] * x1[3] + x2[0] * x2[3]);
        sh_data[tid].JtJ[4] = w * (x1[0] * x1[4] + x2[0] * x2[4]);
        sh_data[tid].JtJ[5] = w * (x1[0] * x1[5] + x2[0] * x2[5]);

        sh_data[tid].JtJ[6] = w * (x1[1] * x1[1] + x2[1] * x2[1]);
        sh_data[tid].JtJ[7] = w * (x1[1] * x1[2] + x2[1] * x2[2]);
        sh_data[tid].JtJ[8] = w * (x1[1] * x1[3] + x2[1] * x2[3]);
        sh_data[tid].JtJ[9] = w * (x1[1] * x1[4] + x2[1] * x2[4]);
        sh_data[tid].JtJ[10] = w * (x1[1] * x1[5] + x2[1] * x2[5]);

        sh_data[tid].JtJ[11] = w * (x1[2] * x1[2] + x2[2] * x2[2]);
        sh_data[tid].JtJ[12] = w * (x1[2] * x1[3] + x2[2] * x2[3]);
        sh_data[tid].JtJ[13] = w * (x1[2] * x1[4] + x2[2] * x2[4]);
        sh_data[tid].JtJ[14] = w * (x1[2] * x1[5] + x2[2] * x2[5]);

        sh_data[tid].JtJ[15] = w * (x1[3] * x1[3] + x2[3] * x2[3]);
        sh_data[tid].JtJ[16] = w * (x1[3] * x1[4] + x2[3] * x2[4]);
        sh_data[tid].JtJ[17] = w * (x1[3] * x1[5] + x2[3] * x2[5]);

        sh_data[tid].JtJ[18] = w * (x1[4] * x1[4] + x2[4] * x2[4]);
        sh_data[tid].JtJ[19] = w * (x1[4] * x1[5] + x2[4] * x2[5]);

        sh_data[tid].JtJ[20] = w * (x1[5] * x1[5] + x2[5] * x2[5]);

        //sh_data[tid].r = dot(source_positions[id] - target_positions[id], ns + nt) * dot(source_positions[id] - target_positions[id], ns + nt);
        //sh_data[tid].r = dot(source_positions[id] - target_positions[id], ns) * dot(source_positions[id] - target_positions[id], ns);
        //sh_data[tid].r = dot(target_positions[id] - source_positions[id], nt) * dot(target_positions[id] - source_positions[id], nt);
        //float3 diff =  target_positions[id] - source_positions[id];
        //sh_data[tid].r = dot(diff, diff);

    }

    __syncthreads();

    reduce<BLOCK_SIZE, MotionTrackingData>(sh_data, tid);

    if(tid == 0)
        atomicAdd(system, sh_data[tid]);
}

} // supersurfel_fusion

#endif // DENSE_REGISTRATION_KERNELS_CUH
