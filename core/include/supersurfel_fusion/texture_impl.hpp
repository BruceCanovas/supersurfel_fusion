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


#ifndef TEXTURE_IMPL_HPP
#define TEXTURE_IMPL_HPP

#include <opencv2/core/cuda.hpp>


namespace supersurfel_fusion
{

template<typename T>
Texture<T>::Texture(const cv::cuda::GpuMat& img, cudaTextureFilterMode filterMode_)
{
    filterMode = filterMode_;
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.data;
    resDesc.res.pitch2D.width = img.cols;
    resDesc.res.pitch2D.height = img.rows;
    resDesc.res.pitch2D.pitchInBytes = img.step;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.normalizedCoords = 0;
    
    cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
}

template<typename T>
Texture<T>::~Texture()
{
    cudaDestroyTextureObject(textureObject);
}

}; //namespace supersurfel_fusion

#endif // TEXTURE_IMPL_HPP
