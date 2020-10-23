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


#ifndef TPS_RGBD_KERNELS_CUH
#define TPS_RGBD_KERNELS_CUH

#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <supersurfel_fusion/matrix_math.cuh>


namespace supersurfel_fusion
{

__global__ void initSuperpixelsRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                           cudaTextureObject_t texImg,
                                           int* indexData,
                                           int* boundaryData,
                                           int cellSize,
                                           int gridSizeX,
                                           int gridSizeY,
                                           cv::Rect roi,
                                           int index_pitch,
                                           int boundary_pitch);

__global__ void initDispCoeffsRansacRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                                unsigned char *inliers_data,
                                                const SuperpixelRGBD* superpixels,
                                                cudaTextureObject_t texDisp,
                                                cudaTextureObject_t texIndex,
                                                float threshold,
                                                cv::Rect roi,
                                                int inliers_pitch);

__global__ void initDispCoeffsRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                          unsigned char *inliers_data,
                                          cudaTextureObject_t texDisp,
                                          cudaTextureObject_t texIndex,
                                          cv::Rect roi,
                                          int inliers_pitch);

__global__ void renderSegmentedImageRGBD_kernel(unsigned char * segmentedData,
                                                const SuperpixelRGBD* superpixels,
                                                const int* indexData,
                                                int width,
                                                int height,
                                                int segmented_pitch,
                                                int index_pitch);


template <int OFFSET_X, int OFFSET_Y, int BLK_SIZE_X, int BLK_SIZE_Y>
__global__ void updateTPSRGBD_kernel(const SuperpixelRGBD* superpixels,
                                     SuperpixelRGBDCoeffs* coeffs,
                                     cudaTextureObject_t texImg,
                                     cudaTextureObject_t texDisp,
                                     int* indexData,
                                     int* boundaryData,
                                     unsigned char* inliersData,
                                     float lambda_pos,
                                     float lambda_bound,
                                     float lambda_size,
                                     float lambda_disp,
                                     float thresh_disp,
                                     int min_size,
                                     cv::Rect roi,
                                     int nbSuperpixels,
                                     int index_pitch,
                                     int boundary_pitch,
                                     int inliers_pitch);

template <int OFFSET_X, int OFFSET_Y, int BLK_SIZE_X, int BLK_SIZE_Y>
__global__ void updateTPSRGB_kernel(const SuperpixelRGBD* superpixels,
                                    SuperpixelRGBDCoeffs* coeffs,
                                    cudaTextureObject_t texImg,
                                    int* indexData,
                                    int* boundaryData,
                                    float lambda_pos,
                                    float lambda_bound,
                                    float lambda_size,
                                    int min_size,
                                    cv::Rect roi,
                                    int nbSuperpixels,
                                    int index_pitch,
                                    int boundary_pitch);

__global__ void mergeTPSRGBCoeffs_kernel(SuperpixelRGBD* superpixels,
                                         const SuperpixelRGBDCoeffs* coeffs,
                                         int nbSuperpixels);

__global__ void mergeTPSRGBDCoeffs_kernel(SuperpixelRGBD* superpixels,
                                          const SuperpixelRGBDCoeffs* coeffs,
                                          int nbSuperpixels);

__global__ void depth2disp32F_kernel(float* disp_data,
                                     const float* depth_data,
                                     int width,
                                     int height,
                                     int disp_pitch,
                                     int depth_pitch);
__global__ void depth2disp16U_kernel(float* disp_data,
                                     const uint16_t* depth_data,
                                     int width,
                                     int height,
                                     int disp_pitch,
                                     int depth_pitch);

__global__ void initRandStates_kernel(curandState* state);

__global__ void initSamples_kernel(float4* samples,
                                   curandState* rand_states,
                                   const SuperpixelRGBD* superpixels,
                                   const cudaTextureObject_t texIndex,
                                   const cudaTextureObject_t texDisp,
                                   const int nbWalks,
                                   const float radius,
                                   cv::Rect roi);

__global__ void evalSamples_kernel(float4* samples,
                                   const cudaTextureObject_t texIndex,
                                   const cudaTextureObject_t texDisp,
                                   float sigma2,
                                   cv::Rect roi,
                                   const int nbSamples);

__global__ void selectSamples_kernel(SuperpixelRGBD* superpixels,
                                     SuperpixelRGBDCoeffs* coeffs,
                                     const float4* samples,
                                     const int nbSuperpixels,
                                     const int nbSamples);

__global__ void renderDepthImage_kernel(float* depthData,
                                        SuperpixelRGBD* superpixels,
                                        const cudaTextureObject_t texIndex,
                                        const cudaTextureObject_t texInliers,
                                        cv::Rect roi,
                                        float scale,
                                        const int width,
                                        const int height,
                                        const int pitch);

__global__ void renderBoundaryImage_kernel(unsigned char * segmentedData,
                                           cudaTextureObject_t texImg,
                                           const int* indexData,
                                           cv::Rect roi,
                                           int segmented_pitch,
                                           int index_pitch);

__global__ void initFilter_kernel(FilterDataRGBD* data,
                                  const SuperpixelRGBD* superpixels,
                                  const int nbSuperpixels);

__global__ void iterateFilter_kernel(FilterDataRGBD* data,
                                     float alpha,
                                     float beta,
                                     float threshold,
                                     const int gridSizeX,
                                     const int gridSizeY);

__global__ void finishFilter_kernel(SuperpixelRGBD* superpixels,
                                    const FilterDataRGBD* data,
                                    const int nbSuperpixels);

__device__ __forceinline__ bool isUnchangeable(const int* indexData,
                                               const int x,
                                               const int y,
                                               const int index_pitch)
{
    const int ptr = y*index_pitch + x;
    const int index = indexData[ptr];
    int jump = 0;

    int i = indexData[ptr-index_pitch-1];
    bool prev = (i==index);
    bool cur = (indexData[ptr-index_pitch] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr-index_pitch+1] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr+1] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr+index_pitch+1] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr+index_pitch] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr+index_pitch-1] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }
    cur = (indexData[ptr-1] == index);
    if(prev!=cur)
    {
        jump++;
        prev = cur;
    }

    return jump>2;
}

template <int OFFSET_X, int OFFSET_Y, int BLK_SIZE_X, int BLK_SIZE_Y>
__global__ void updateTPSRGBD_kernel(SuperpixelRGBD* superpixels,
                                     SuperpixelRGBDCoeffs* coeffs,
                                     cudaTextureObject_t texImg,
                                     cudaTextureObject_t texDisp,
                                     int* indexData,
                                     int* boundaryData,
                                     unsigned char* inliersData,
                                     float lambda_pos,
                                     float lambda_bound,
                                     float lambda_size,
                                     float lambda_disp,
                                     float thresh_disp,
                                     int min_size,
                                     cv::Rect roi,
                                     int nbSuperpixels,
                                     int index_pitch,
                                     int boundary_pitch,
                                     int inliers_pitch)
{
    const int x0 = 2*blockIdx.x *blockDim.x - 1;
    const int y0 = 2*blockIdx.y *blockDim.y + OFFSET_Y - 1;

    if(x0+1>= roi.width || y0+1>=roi.height)
        return;

    const int raw_x = blockIdx.x *blockDim.x + threadIdx.x;
    const int raw_y = blockIdx.y *blockDim.y + threadIdx.y;

    const int x = 2*raw_x + (raw_x+OFFSET_X)%2;
    const int y = 2*raw_y + OFFSET_Y;

    const int sh_x = 2*threadIdx.x+1+(raw_x+OFFSET_X)%2;
    const int sh_y = 2*threadIdx.y+1;
    const int sh_w = 2*BLK_SIZE_X + 2;
    const int sh_h = 2*BLK_SIZE_Y + 2;

    __shared__ int shIndex[sh_w*sh_h];

    #pragma unroll
    for(int i=threadIdx.y; i<sh_h; i+=blockDim.y)
    {
        int yy = y0 + i;
        #pragma unroll
        for(int j=threadIdx.x; j<sh_w; j+=blockDim.x)
        {
            int xx = x0 + j;
            if(xx >=0 && xx<roi.width
                    && yy>=0 && yy<roi.height)
            {
                shIndex[i*sh_w+j] = indexData[yy*index_pitch+xx];
            }else{
                shIndex[i*sh_w+j] = -1;
            }
        }
    }

    __syncthreads();

    if(x>=roi.width || y>=roi.height)
        return;


    //   indexData[y*index_pitch+x] = shIndex[sh_y*sh_w+sh_x + 1];
    //   return;

    int bounds = boundaryData[y*boundary_pitch+x];

    int ptr = sh_y*sh_w + sh_x;
    int index = shIndex[ptr];
    int new_index = index;

    SuperpixelRGBD prev_sp = superpixels[index];

    float disp = tex2D<float>(texDisp, x+roi.x, y+roi.y);
    unsigned char prev_inlier = inliersData[y*inliers_pitch+x];

    float dp = prev_sp.theta_b.x*x+prev_sp.theta_b.y*y+prev_sp.theta_b.z;
    float disp_energy = (dp-disp)*(dp-disp);

    unsigned char inlier = 0xff;

    if(!isfinite(disp_energy) || disp_energy>thresh_disp || dp < 0.f)
    {
        disp_energy = thresh_disp;
        inlier = 0;
    }

    if(bounds && !isUnchangeable(shIndex, sh_x, sh_y, sh_w))
    {
        uchar4 color = tex2D<uchar4>(texImg, x+roi.x, y+roi.y);

        float3 color3 = make_float3(color.x, color.y, color.z);
        float2 pos = make_float2(x, y);

        const float size = prev_sp.size.x;

        float s = size / (size-1.f);

        float2 dpos = make_float2(s * (pos.x - prev_sp.xy_rg.x),
                                  s * (pos.y - prev_sp.xy_rg.y));

        float3 dcolor = make_float3(s * (color3.x - prev_sp.xy_rg.z),
                                    s * (color3.y - prev_sp.xy_rg.w),
                                    s * (color3.z - prev_sp.theta_b.w));

        float dsize = size - min_size;


        float best_energy = dot(dcolor, dcolor)
                + lambda_pos*dot(dpos, dpos)
                + lambda_disp*disp_energy
                - lambda_size*min(dsize, 0.f)
                + lambda_bound * bounds;

        const int neighbors[8] = {0,-1,-1,0,1,0,0,1};

        #pragma unroll
        for(int k=0; k<4; k++)
        {
            int i_n = shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]];
            if(i_n==-1 || i_n==index)
                continue;

            SuperpixelRGBD n_sp = superpixels[i_n];
            float2 dpos = pos - *((float2*)&n_sp.xy_rg);
            float3 dcolor = color3 - make_float3(n_sp.xy_rg.z, n_sp.xy_rg.w, n_sp.theta_b.w);

            float dsize = n_sp.size.x + 1.f - min_size;

            float dp = n_sp.theta_b.x*x+n_sp.theta_b.y*y+n_sp.theta_b.z;
            float disp_energy = (dp-disp)*(dp-disp);

            unsigned char n_inlier = 0xff;
            if(!isfinite(disp_energy) || disp_energy>thresh_disp || dp<0.f)
            {
                disp_energy = thresh_disp;
                n_inlier = 0;
            }

            int b=0;
            for(int k=0; k<4; k++)
            {
                if(shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]]!=i_n)
                {
                    b++;
                }
            }

            float energy = dot(dcolor, dcolor)
                    + lambda_pos*dot(dpos, dpos)
                    + lambda_disp*disp_energy
                    - lambda_size*min(dsize, 0.f)
                    + lambda_bound*b;

            if(energy<best_energy)
            {
                best_energy = energy;
                new_index = i_n;
                inlier = n_inlier;
            }
        }

        __syncthreads();

        if(new_index != index)
        {
            int b = 0;
            #pragma unroll
            for(int k=0; k<4; k++)
            {
                int i_n = shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]];
                if(i_n == new_index)
                {
                    //atomicAdd(&boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]],-1);
                    boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]]--;

                }else{
                    b++;
                    if(i_n == index)
                    {
                        //atomicAdd(&boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]],1);
                        boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]]++;
                    }
                }
            }

            __syncthreads();
            boundaryData[y*boundary_pitch+x] = b;

            atomicAdd(&(coeffs[index].x), (float) -x);
            atomicAdd(&(coeffs[index].y), (float) -y);
            atomicAdd(&(coeffs[index].r), (float) -color.x);
            atomicAdd(&(coeffs[index].g), (float) -color.y);
            atomicAdd(&(coeffs[index].b), (float) -color.z);
            atomicAdd(&(coeffs[index].n), (float) -1.f);

            atomicAdd(&(coeffs[new_index].x), (float) x);
            atomicAdd(&(coeffs[new_index].y), (float) y);
            atomicAdd(&(coeffs[new_index].r), (float) color.x);
            atomicAdd(&(coeffs[new_index].g), (float) color.y);
            atomicAdd(&(coeffs[new_index].b), (float) color.z);
            atomicAdd(&(coeffs[new_index].n), (float) 1.f);

            indexData[y*index_pitch+x] =  new_index;
        }
    }

    if(inlier & (!prev_inlier || index!=new_index))
    {
        atomicAdd(&(coeffs[new_index].dx), (float) x);
        atomicAdd(&(coeffs[new_index].dy), (float) y);
        atomicAdd(&(coeffs[new_index].dxx), (float) x*x);
        atomicAdd(&(coeffs[new_index].dyy), (float) y*y);
        atomicAdd(&(coeffs[new_index].dxy), (float) x*y);
        atomicAdd(&(coeffs[new_index].dxd), (float) x*disp);
        atomicAdd(&(coeffs[new_index].dyd), (float) y*disp);
        atomicAdd(&(coeffs[new_index].dd), (float) disp);
        atomicAdd(&(coeffs[new_index].dn), (float) 1.f);
    }

    if(prev_inlier && (!inlier || (inlier && index!=new_index)))
    {
        atomicAdd(&(coeffs[index].dx), (float) -x);
        atomicAdd(&(coeffs[index].dy), (float) -y);
        atomicAdd(&(coeffs[index].dxx), (float) -x*x);
        atomicAdd(&(coeffs[index].dyy), (float) -y*y);
        atomicAdd(&(coeffs[index].dxy), (float) -x*y);
        atomicAdd(&(coeffs[index].dxd), (float) -x*disp);
        atomicAdd(&(coeffs[index].dyd), (float) -y*disp);
        atomicAdd(&(coeffs[index].dd), (float) -disp);
        atomicAdd(&(coeffs[index].dn), (float) -1.f);
    }

    if(inlier!=prev_inlier)
    {
        inliersData[y*inliers_pitch+x] = inlier;
    }

}

template <int OFFSET_X, int OFFSET_Y, int BLK_SIZE_X, int BLK_SIZE_Y>
__global__ void updateTPSRGB_kernel(SuperpixelRGBD* superpixels,
                                    SuperpixelRGBDCoeffs* coeffs,
                                    cudaTextureObject_t texImg,
                                    int* indexData,
                                    int* boundaryData,
                                    float lambda_pos,
                                    float lambda_bound,
                                    float lambda_size,
                                    int min_size,
                                    cv::Rect roi,
                                    int nbSuperpixels,
                                    int index_pitch,
                                    int boundary_pitch)
{
    const int x0 = 2*blockIdx.x *blockDim.x - 1;
    const int y0 = 2*blockIdx.y *blockDim.y + OFFSET_Y - 1;

    if(x0+1>= roi.width || y0+1>=roi.height)
        return;

    const int raw_x = blockIdx.x *blockDim.x + threadIdx.x;
    const int raw_y = blockIdx.y *blockDim.y + threadIdx.y;

    const int x = 2*raw_x + (raw_x+OFFSET_X)%2;
    const int y = 2*raw_y + OFFSET_Y;

    const int sh_x = 2*threadIdx.x+1+(raw_x+OFFSET_X)%2;
    const int sh_y = 2*threadIdx.y+1;
    const int sh_w = 2*BLK_SIZE_X + 2;
    const int sh_h = 2*BLK_SIZE_Y + 2;

    __shared__ int shIndex[sh_w*sh_h];

    #pragma unroll
    for(int i=threadIdx.y; i<sh_h; i+=blockDim.y)
    {
        int yy = y0 + i;
        #pragma unroll
        for(int j=threadIdx.x; j<sh_w; j+=blockDim.x)
        {
            int xx = x0 + j;
            if(xx >=0 && xx<roi.width
                    && yy>=0 && yy<roi.height)
            {
                shIndex[i*sh_w+j] = indexData[yy*index_pitch+xx];
            }else{
                shIndex[i*sh_w+j] = -1;
            }
        }
    }

    __syncthreads();

    if(x>=roi.width || y>=roi.height)
        return;


    //   indexData[y*index_pitch+x] = shIndex[sh_y*sh_w+sh_x + 1];
    //   return;

    int bounds = boundaryData[y*boundary_pitch+x];

    int ptr = sh_y*sh_w + sh_x;
    int index = shIndex[ptr];
    int new_index = index;

    SuperpixelRGBD prev_sp = superpixels[index];

    if(bounds && !isUnchangeable(shIndex, sh_x, sh_y, sh_w))
    {
        uchar4 color = tex2D<uchar4>(texImg, x+roi.x, y+roi.y);

        float3 color3 = make_float3(color.x, color.y, color.z);
        float2 pos = make_float2(x, y);

        const float size = prev_sp.size.x;

        float s = size / (size-1.f);

        float2 dpos = make_float2(s * (pos.x - prev_sp.xy_rg.x),
                                  s * (pos.y - prev_sp.xy_rg.y));

        float3 dcolor = make_float3(s * (color3.x - prev_sp.xy_rg.z),
                                    s * (color3.y - prev_sp.xy_rg.w),
                                    s * (color3.z - prev_sp.theta_b.w));

        float dsize = size - min_size;


        float best_energy = dot(dcolor, dcolor)
                + lambda_pos*dot(dpos, dpos)
                - lambda_size*min(dsize, 0.f)
                + lambda_bound * bounds;

        const int neighbors[8] = {0,-1,-1,0,1,0,0,1};

        #pragma unroll
        for(int k=0; k<4; k++)
        {
            int i_n = shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]];
            if(i_n==-1 || i_n==index)
                continue;

            SuperpixelRGBD n_sp = superpixels[i_n];
            float2 dpos = pos - *((float2*)&n_sp.xy_rg);
            float3 dcolor = color3 - make_float3(n_sp.xy_rg.z, n_sp.xy_rg.w, n_sp.theta_b.w);

            float dsize = n_sp.size.x + 1.f - min_size;

            int b=0;
            for(int k=0; k<4; k++)
            {
                if(shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]]!=i_n)
                {
                    b++;
                }
            }

            float energy = dot(dcolor, dcolor)
                    + lambda_pos*dot(dpos, dpos)
                    - lambda_size*min(dsize, 0.f)
                    + lambda_bound*b;

            if(energy<best_energy)
            {
                best_energy = energy;
                new_index = i_n;
            }
        }

        __syncthreads();

        if(new_index != index)
        {
            int b = 0;
            #pragma unroll
            for(int k=0; k<4; k++)
            {
                int i_n = shIndex[(sh_y+neighbors[2*k+1])*sh_w+sh_x+neighbors[2*k]];
                if(i_n == new_index)
                {
                    //atomicAdd(&boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]],-1);
                    boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]]--;

                }else{
                    b++;
                    if(i_n == index)
                    {
                        //atomicAdd(&boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]],1);
                        boundaryData[(y+neighbors[2*k+1])*boundary_pitch+x+neighbors[2*k]]++;
                    }
                }
            }

            __syncthreads();
            boundaryData[y*boundary_pitch+x] = b;

            atomicAdd(&(coeffs[index].x), (float) -x);
            atomicAdd(&(coeffs[index].y), (float) -y);
            atomicAdd(&(coeffs[index].r), (float) -color.x);
            atomicAdd(&(coeffs[index].g), (float) -color.y);
            atomicAdd(&(coeffs[index].b), (float) -color.z);
            atomicAdd(&(coeffs[index].n), (float) -1.f);

            atomicAdd(&(coeffs[new_index].x), (float) x);
            atomicAdd(&(coeffs[new_index].y), (float) y);
            atomicAdd(&(coeffs[new_index].r), (float) color.x);
            atomicAdd(&(coeffs[new_index].g), (float) color.y);
            atomicAdd(&(coeffs[new_index].b), (float) color.z);
            atomicAdd(&(coeffs[new_index].n), (float) 1.f);

            indexData[y*index_pitch+x] =  new_index;
        }
    }
}

} // namespace supersurfel_fusion

#endif // TPS_RGBD_KERNELS_CUH
