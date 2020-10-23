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


#ifndef FERNS_KERNELS_CUH
#define FERNS_KERNELS_CUH

#include <supersurfel_fusion/ferns_types.hpp>


namespace supersurfel_fusion
{

__global__ void initFerns_kernel(Fern* ferns, float maxDepth, uint32_t width, uint32_t height, int n);

__global__ void computeCodes_kernel(uint8_t* codes,
                                    const Fern* ferns,
                                    const cudaTextureObject_t texColor,
                                    const cudaTextureObject_t texDepth,
                                    int n);

} // supersurfel_fusion

#endif // FERNS_KERNELS_CUH
