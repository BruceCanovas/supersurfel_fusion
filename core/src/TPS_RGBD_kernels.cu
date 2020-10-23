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


#include <supersurfel_fusion/TPS_RGBD_kernels.cuh>
#include <stdio.h>


namespace supersurfel_fusion
{

__device__ inline bool solvePlaneEquations(float4& theta,
                                           const float x1, const float y1, const float z1, const float d1,
                                           const float x2, const float y2, const float z2, const float d2,
                                           const float x3, const float y3, const float z3, const float d3)
{
    const float epsilonValue = 1e-20;

    float denominatorA = (x1*z2 - x2*z1)*(y2*z3 - y3*z2) - (x2*z3 - x3*z2)*(y1*z2 - y2*z1);
    if (!isfinite(denominatorA) && denominatorA < epsilonValue) {
        //theta.x = 0.0;
        //theta.y = 0.0;
        //theta.z = -1.0;
        return false;
    }

    theta.x = ((z2*d1 - z1*d2)*(y2*z3 - y3*z2) - (z3*d2 - z2*d3)*(y1*z2 - y2*z1))/denominatorA;

    float denominatorB = y1*z2 - y2*z1;
    if (denominatorB > epsilonValue) {
        theta.y = (z2*d1 - z1*d2 - theta.x*(x1*z2 - x2*z1))/denominatorB;
    } else {
        denominatorB = y2*z3 - y3*z2;
        theta.y = (z3*d2 - z2*d3 - theta.x*(x2*z3 - x3*z2))/denominatorB;
    }
    if (z1 > epsilonValue) {
        theta.z = (d1 - theta.x*x1 - theta.y*y1)/z1;
    } else if (z2 > epsilonValue) {
        theta.z = (d2 - theta.x*x2 - theta.y*y2)/z2;
    } else {
        theta.z = (d3 - theta.x*x3 - theta.y*y3)/z3;
    }
    return true;
}

__global__ void initSuperpixelsRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                           cudaTextureObject_t texImg,
                                           int* indexData,
                                           int* boundaryData,
                                           int cellSize,
                                           int gridSizeX,
                                           int gridSizeY,
                                           cv::Rect roi,
                                           int index_pitch,
                                           int boundary_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=roi.width  || y >= roi.height)
        return;

    int ycell = y / cellSize;
    int xcell = x / cellSize;
    int index = gridSizeX * ycell + xcell;

    indexData[y*index_pitch + x] = index;

    int xmod = (x+1) % cellSize;
    int ymod = (y+1) % cellSize;

    //int nbSuperpixels = gridSizeX*gridSizeY;

    int b=0;

    if(xmod <=1)
    {
        b++;
    }
    if(ymod <=1)
    {
        b++;
    }

    boundaryData[y*boundary_pitch + x] = b;

    uchar4 color = tex2D<uchar4>(texImg, x + roi.x, y + roi.y);

    atomicAdd(&(coeffs[index].x), (float) x);
    atomicAdd(&(coeffs[index].y), (float) y);
    atomicAdd(&(coeffs[index].r), (float) color.x);
    atomicAdd(&(coeffs[index].g), (float) color.y);
    atomicAdd(&(coeffs[index].b), (float) color.z);
    atomicAdd(&(coeffs[index].n), 1.f);
}

__global__ void initDispCoeffsRansacRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                                unsigned char *inliers_data,
                                                const SuperpixelRGBD* superpixels,
                                                cudaTextureObject_t texDisp,
                                                cudaTextureObject_t texIndex,
                                                float threshold,
                                                cv::Rect roi,
                                                int inliers_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=roi.width  || y >= roi.height)
        return;

    int index = tex2D<int>(texIndex, x, y);
    float d = tex2D<float>(texDisp, x+roi.x, y+roi.y);

    unsigned char inlier = 0;

    if(isfinite(d))
    {
        float4 theta = superpixels[index].theta_b;

        float dp = theta.x*x+theta.y*y+theta.z;

        float dd = (dp-d)*(dp-d);
        if(isfinite(dd) && dd < threshold && dp>0.f)
        {
            inlier = 0xff;
            atomicAdd(&(coeffs[index].dx), (float) x);
            atomicAdd(&(coeffs[index].dy), (float) y);
            atomicAdd(&(coeffs[index].dxx), (float)(x*x));
            atomicAdd(&(coeffs[index].dyy), (float)(y*y));
            atomicAdd(&(coeffs[index].dxy), (float)(x*y));
            atomicAdd(&(coeffs[index].dxd), (float)(x*d));
            atomicAdd(&(coeffs[index].dyd), (float)(y*d));
            atomicAdd(&(coeffs[index].dd), d);
            atomicAdd(&(coeffs[index].dn), 1.f);
        }
    }

    inliers_data[y*inliers_pitch+x] = inlier;
}

__global__ void initDispCoeffsRGBD_kernel(SuperpixelRGBDCoeffs* coeffs,
                                          unsigned char *inliers_data,
                                          cudaTextureObject_t texDisp,
                                          cudaTextureObject_t texIndex,
                                          cv::Rect roi,
                                          int inliers_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=roi.width  || y >= roi.height)
        return;

    int index = tex2D<int>(texIndex, x, y);
    float d = tex2D<float>(texDisp, x+roi.x, y+roi.y);

    unsigned char inlier = 0;

    if(isfinite(d))
    {
        inlier = 0xff;
        atomicAdd(&(coeffs[index].dx), (float) x);
        atomicAdd(&(coeffs[index].dy), (float) y);
        atomicAdd(&(coeffs[index].dxx), (float)(x*x));
        atomicAdd(&(coeffs[index].dyy), (float)(y*y));
        atomicAdd(&(coeffs[index].dxy), (float)(x*y));
        atomicAdd(&(coeffs[index].dxd), (float)(x*d));
        atomicAdd(&(coeffs[index].dyd), (float)(y*d));
        atomicAdd(&(coeffs[index].dd), d);
        atomicAdd(&(coeffs[index].dn), 1.f);
    }

    inliers_data[y*inliers_pitch+x] = inlier;
}


__global__ void renderSegmentedImageRGBD_kernel(unsigned char * segmentedData,
                                                const SuperpixelRGBD* superpixels,
                                                const int* indexData,
                                                int width,
                                                int height,
                                                int segmented_pitch,
                                                int index_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=width  || y >= height)
        return;

    int index = indexData[y*index_pitch+x];

    if(index < 0 )
    {
        segmentedData[y*segmented_pitch+3*x] = 255;
        segmentedData[y*segmented_pitch+3*x+1] = 255;
        segmentedData[y*segmented_pitch+3*x+2] = 255;
        return;
    }

    const SuperpixelRGBD& sp = superpixels[index];

    segmentedData[y*segmented_pitch+3*x] = (unsigned char) sp.xy_rg.z;
    segmentedData[y*segmented_pitch+3*x+1] = (unsigned char) sp.xy_rg.w;
    segmentedData[y*segmented_pitch+3*x+2] = (unsigned char) sp.theta_b.w;
}

__global__ void mergeTPSRGBCoeffs_kernel(SuperpixelRGBD* superpixels,
                                         const SuperpixelRGBDCoeffs* coeffs,
                                         int nbSuperpixels)
{
    int k = blockIdx.x *blockDim.x + threadIdx.x;
    if(k>=nbSuperpixels)
        return;

    SuperpixelRGBD sp;
    SuperpixelRGBDCoeffs c = coeffs[k];
    sp.xy_rg.x = c.x/c.n;
    sp.xy_rg.y = c.y/c.n;
    sp.xy_rg.z = c.r/c.n;
    sp.xy_rg.w = c.g/c.n;
    sp.theta_b.w = c.b/c.n;
    sp.size.x = c.n;

    superpixels[k] = sp;
}

__global__ void mergeTPSRGBDCoeffs_kernel(SuperpixelRGBD* superpixels,
                                          const SuperpixelRGBDCoeffs* coeffs,
                                          int nbSuperpixels)
{
    int k = blockIdx.x *blockDim.x + threadIdx.x;
    if(k>=nbSuperpixels)
        return;

    SuperpixelRGBD sp;
    SuperpixelRGBDCoeffs c = coeffs[k];
    sp.xy_rg.x = c.x/c.n;
    sp.xy_rg.y = c.y/c.n;
    sp.xy_rg.z = c.r/c.n;
    sp.xy_rg.w = c.g/c.n;
    sp.theta_b.w = c.b/c.n;
    sp.size.x = c.n;

    float4 theta;
    if(!solvePlaneEquations(theta,
                            c.dxx, c.dxy, c.dx, c.dxd,
                            c.dxy, c.dyy, c.dy, c.dyd,
                            c.dx, c.dy, c.dn, c.dd))
    {
        theta.x = 0.f;
        theta.y = 0.f;
        theta.z = __int_as_float(0xFFE00000);
    }
    sp.theta_b.x = theta.x;
    sp.theta_b.y = theta.y;
    sp.theta_b.z = theta.z;

    superpixels[k] = sp;
}

__global__ void depth2disp32F_kernel(float* disp_data,
                                     const float* depth_data,
                                     int width,
                                     int height,
                                     int disp_pitch,
                                     int depth_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=width  || y >= height)
        return;

    float depth = depth_data[y*depth_pitch+x];
    //if(depth>0.1f)
    disp_data[y*disp_pitch+x] = 1.f / depth;
    //else
    //disp_data[y*disp_pitch+x] = __int_as_float(0xFFE00000);
}

__global__ void depth2disp16U_kernel(float* disp_data,
                                     const uint16_t* depth_data,
                                     int width,
                                     int height,
                                     int disp_pitch,
                                     int depth_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=width  || y >= height)
        return;

    float depth = depth_data[y*depth_pitch+x]/1000.f;
    //if(depth>0.1f)
    disp_data[y*disp_pitch+x] = 1.f / depth;
    //else
    //disp_data[y*disp_pitch+x] = __int_as_float(0xFFE00000);
}

__global__ void initRandStates_kernel(curandState* state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

__global__ void initSamples_kernel(float4* samples,
                                   curandState* rand_states,
                                   const SuperpixelRGBD* superpixels,
                                   const cudaTextureObject_t texIndex,
                                   const cudaTextureObject_t texDisp,
                                   const int nbWalks,
                                   const float radius,
                                   cv::Rect roi)
{
    int index = blockIdx.x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    curandState rnd = rand_states[idx];

    float cx = superpixels[index].xy_rg.x;
    float cy = superpixels[index].xy_rg.y;

    float x = cx;
    float y = cy;

    int i = tex2D<int>(texIndex, x, y);
    int k=0;
    while( i != index && k++<10)
    {
        x = cx + radius*2.*(curand_uniform(&rnd)-1.f);
        y = cy + radius*2.*(curand_uniform(&rnd)-1.f);
        i = tex2D<int>(texIndex, x, y);
    }

    __syncthreads();

    float dx[4] = {-1.f, 0.f, 1.f, 0.f};
    float dy[4] = {0., -1.f, 0.f, 1.f};

    float3 xyd[3];

    float d = tex2D<float>(texDisp, x+roi.x, y+roi.y);
    xyd[0] = xyd[1] = xyd[2] = make_float3(x, y, d);

    #pragma unroll
    for(int j=0; j<3; j++)
    {
        for(int k=0; k<nbWalks; k++)
        {
            int dir = curand(&rnd) & 3;
            float next_x = x+dx[dir];
            float next_y = y+dy[dir];
            i = tex2D<int>(texIndex, x, y);
            if(i==index && next_x >=0 && next_x<roi.width && next_y>=0 && next_y<roi.height)
            {
                x = next_x;
                y = next_y;

                float d = tex2D<float>(texDisp, x+roi.x, y+roi.y);
                if(isfinite(d))
                {
                    xyd[j]=make_float3(x, y, d);
                }
            }

        }
    }

    float4 sample;
    if(!solvePlaneEquations(sample,
                            xyd[0].x, xyd[0].y, 1.f, xyd[0].z,
                            xyd[1].x, xyd[1].y, 1.f, xyd[1].z,
                            xyd[2].x, xyd[2].y, 1.f, xyd[2].z))
    {
        sample.x = 0.f;
        sample.y = 0.f;
        sample.z = xyd[2].z;
    }

    sample.w = 0.f;

    samples[idx] = sample;
    rand_states[idx] = rnd;
}

__global__ void evalSamples_kernel(float4* samples,
                                   const cudaTextureObject_t texIndex,
                                   const cudaTextureObject_t texDisp,
                                   float sigma2,
                                   cv::Rect roi,
                                   const int nbSamples)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=roi.width || y >= roi.height)
        return;

    int index = tex2D<int>(texIndex, x, y);
    float d = tex2D<float>(texDisp, x + roi.x, y + roi.y);

    for(int k=0; k<nbSamples; k++)
    {
        float4 theta = samples[index*nbSamples+k];
        if(isfinite(theta.z))
        {
            float dp = theta.x*x+theta.y*y+theta.z;

            float dd = (d-dp)*(d-dp);
            //atomicAdd(&samples[index*nbSamples+k].w, expf(-0.5f*dd/sigma2));
            if(dd<sigma2)
                atomicAdd(&samples[index*nbSamples+k].w, 1.f);

        }
    }
}

__global__ void selectSamples_kernel(SuperpixelRGBD* superpixels,
                                     SuperpixelRGBDCoeffs* coeffs,
                                     const float4* samples,
                                     const int nbSuperpixels,
                                     const int nbSamples)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx >= nbSuperpixels)
        return;

    float4 best = make_float4(0.f, 0.f, 0.f, 0.f);
    for(int k=0; k<nbSamples; k++)
    {
        float4 theta = samples[idx*nbSamples+k];
        if(theta.w > best.w)
            best = theta;
    }

    superpixels[idx].theta_b.x = best.x;
    superpixels[idx].theta_b.y = best.y;
    superpixels[idx].theta_b.z = best.z;

    coeffs[idx].dx = 0.f;
    coeffs[idx].dy = 0.f;
    coeffs[idx].dxx = 0.f;
    coeffs[idx].dyy = 0.f;
    coeffs[idx].dxy = 0.f;
    coeffs[idx].dxd = 0.f;
    coeffs[idx].dyd = 0.f;
    coeffs[idx].dd = 0.f;
    coeffs[idx].dn = 0.f;
}

__global__ void renderDepthImage_kernel(float* depthData,
                                        SuperpixelRGBD* superpixels,
                                        const cudaTextureObject_t texIndex,
                                        const cudaTextureObject_t texInliers,
                                        cv::Rect roi,
                                        float scale,
                                        const int width,
                                        const int height,
                                        const int pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;
    if(x>=width || y >=height)
    {
        return;
    }

    if(x<roi.x || x>=roi.y+roi.width || y<roi.y || y >= roi.y+roi.height)
    {
        depthData[y*pitch+x] = 0.f;
        return;
    }

    float xx = x-roi.x;
    float yy = y-roi.y;

    unsigned char inlier = tex2D<unsigned char>(texInliers, xx, yy);

    if(true || inlier)
    {
        int index = tex2D<int>(texIndex, xx, yy);
        float4 theta = superpixels[index].theta_b;

        float disp = xx*theta.x+yy*theta.y+theta.z;

        depthData[y*pitch+x] = scale/disp;
    }else{
        depthData[y*pitch+x] = 0.f;
    }
}

__global__ void initFilter_kernel(FilterDataRGBD* data,
                                  const SuperpixelRGBD* superpixels,
                                  const int nbSuperpixels)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx >= nbSuperpixels)
        return;

    SuperpixelRGBD sp = superpixels[idx];
    float3 X = make_float3(sp.xy_rg.x*sp.theta_b.x + sp.xy_rg.y*sp.theta_b.y + sp.theta_b.z,
                           sp.theta_b.x,
                           sp.theta_b.y);
    data[idx].X = X;
    data[idx].Z = X;
    data[idx].px = sp.xy_rg.x;
    data[idx].py = sp.xy_rg.y;
}

__global__ void finishFilter_kernel(SuperpixelRGBD* superpixels,
                                    const FilterDataRGBD* data,
                                    const int nbSuperpixels)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(idx >= nbSuperpixels)
        return;

    float3 X = data[idx].X;
    SuperpixelRGBD sp = superpixels[idx];

    sp.theta_b.x = X.y;
    sp.theta_b.y = X.z;
    sp.theta_b.z = X.x-sp.xy_rg.x*X.y-sp.xy_rg.y*X.z;

    superpixels[idx].theta_b = sp.theta_b;
}

__global__ void iterateFilter_kernel(FilterDataRGBD* data,
                                     float alpha,
                                     float beta,
                                     float threshold,
                                     const int gridSizeX,
                                     const int gridSizeY)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=gridSizeX || y >=gridSizeY)
    {
        return;
    }
    int idx = y*gridSizeX + x;

    Cov3 A = make_cov3(alpha, 0.f, 0.f, alpha, 0.f, alpha);
    Mat33 B = make_mat33(0.f, 0.f, 0.f,
                         0.f, 0.f, 0.f,
                         0.f, 0.f, 0.f);

    FilterDataRGBD node_i = data[idx];

    float3 R = alpha*node_i.Z;

    const int v[4] = {-1,0,0,1};
    const int u[4] = {0,-1,1,0};


#pragma unroll
    for(int j=0; j<4; j++)
    {
        int yy = y + v[j];
        int xx = x + u[j];
        if(yy>=0 && yy<gridSizeY
                && xx>=0 && x<gridSizeX)
        {
            FilterDataRGBD node_j = data[yy*gridSizeX+xx];
            float3 Xj = node_j.X;

            float dx = node_i.px-node_j.px;
            float dy = node_i.py-node_j.py;

            float dz = node_i.X.x - (Xj.x/*+dx*Xj.y+dy*Xj.z*/);

            if(isfinite(dz) && dz*dz < threshold*threshold)
            {
                A.xx += beta*2.f;
                A.xy += -beta*dx;
                A.xz += -beta*dy;
                A.yy += beta*(2.f+dx*dx);
                A.yz += beta*(dx*dy);
                A.zz += beta*(2.f+dy*dy);

                R.x += beta*(2.f*Xj.x+dx*Xj.y+dy*Xj.z);
                R.y += beta*(-dx*Xj.x+2.f*Xj.y);
                R.z += beta*(-dy*Xj.x+2.f*Xj.z);
            }
        }
    }

    Cov3 A_1;
    if(inverse(A, A_1))
    {
        data[idx].X = A_1*R;
    }
}

__global__ void renderBoundaryImage_kernel(unsigned char * segmentedData,
                                           cudaTextureObject_t texImg,
                                           const int* indexData,
                                           cv::Rect roi,
                                           int segmented_pitch,
                                           int index_pitch)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x>=roi.width  || y >= roi.height)
        return;
    int index = indexData[y*index_pitch+x];

    if(x<roi.width-1 && y<roi.height-1 &&
            (indexData[y*index_pitch+x+1]!=index
             || indexData[(y+1)*index_pitch+x+1]!=index))
    {
        segmentedData[y*segmented_pitch+3*x] = 255;
        segmentedData[y*segmented_pitch+3*x+1] = 255;
        segmentedData[y*segmented_pitch+3*x+2] = 255;
    }else{
        uchar4 color = tex2D<uchar4>(texImg, x+roi.x, y+roi.y);
        segmentedData[y*segmented_pitch+3*x] = 0.8f*color.z;
        segmentedData[y*segmented_pitch+3*x+1] = 0.8f*color.y;
        segmentedData[y*segmented_pitch+3*x+2] = 0.8f*color.x;
    }
}

} // namespace supersurfel_fusion
