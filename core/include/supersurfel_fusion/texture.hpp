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


#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


namespace supersurfel_fusion
{

template <typename T>
class Texture
{

private:
    cudaTextureObject_t textureObject;
    cudaTextureFilterMode filterMode;

public:
    Texture(const cv::cuda::GpuMat& img, cudaTextureFilterMode filterMode_=cudaFilterModePoint);
    
    ~Texture();
    
    inline cudaTextureObject_t& getTextureObject(){ return textureObject; }

}; // class Texture

} // namespace supersurfel_fusion

#include <supersurfel_fusion/texture_impl.hpp>


#endif // TEXTURE_HPP
