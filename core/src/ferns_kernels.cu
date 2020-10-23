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


#include <supersurfel_fusion/ferns_kernels.cuh>
#include <curand_kernel.h>


namespace supersurfel_fusion
{

__global__ void initFerns_kernel(Fern* ferns, float maxDepth, uint32_t width, uint32_t height, int n)
{
    int k = blockIdx.x *blockDim.x + threadIdx.x;

    if(k>=n)
        return;

    curandState state;
    curand_init(1234, k, 0, &state);

    Fern f;
    f.pos = make_uint2(curand(&state) % width,
                       curand(&state) % height);
    f.rgb = make_uchar3(curand(&state) & 0xff,
                        curand(&state) & 0xff,
                        curand(&state) & 0xff);
    f.depth = maxDepth * curand_uniform(&state);

    ferns[k] = f;
}

__global__ void computeCodes_kernel(uint8_t* codes,
                                    const Fern* ferns,
                                    const cudaTextureObject_t texColor,
                                    const cudaTextureObject_t texDepth,
                                    int n)
{
    int k = blockIdx.x *blockDim.x + threadIdx.x;
    if(k>=n)
        return;

    Fern f = ferns[k];

    uchar4 rgba = tex2D<uchar4>(texColor, f.pos.x, f.pos.y);
    float depth = tex2D<float>(texDepth, f.pos.x, f.pos.y);

    uint8_t r = 0;
    r |= rgba.x > f.rgb.x ? 1 : 0;
    r |= rgba.y > f.rgb.y ? 2 : 0;
    r |= rgba.z > f.rgb.z ? 4 : 0;
    r |= depth > f.depth ? 8 : 0;

    codes[k] = r;
}

} // supersurfel_fusion
