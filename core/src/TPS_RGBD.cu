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


#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <supersurfel_fusion/TPS_RGBD_kernels.cuh>
#include <supersurfel_fusion/cuda_error_check.h>
#include <thrust/fill.h>
#include <iostream>

using namespace std;
using namespace cv;


namespace supersurfel_fusion
{

#define BLOCK_SIZE 256

TPS_RGBD::TPS_RGBD()
    : cellSize(20),
      lambdaPos(50.0f),
      lambdaBound(1e3),
      lambdaSize(1e4),
      lambdaDisp(1e8),
      threshDisp(1e-4),
      nbIters(10),
      useRansac(true),
      nbSamples(16),
      filterIter(4),
      filterAlpha(0.1f),
      filterBeta(0.5f),
      filterThresh(0.02f)
{
}

TPS_RGBD::TPS_RGBD(int cell_size,
                   float lambda_pos,
                   float lambda_bound,
                   float lambda_size,
                   float lambda_disp,
                   float thresh_disp,
                   int nb_iters,
                   bool use_ransac,
                   int nb_samples,
                   int filter_iter,
                   float filter_alpha,
                   float filter_beta,
                   float filter_thresh)
    : cellSize(cell_size),
      lambdaPos(lambda_pos),
      lambdaBound(lambda_bound),
      lambdaSize(lambda_size),
      lambdaDisp(lambda_disp),
      threshDisp(thresh_disp),
      nbIters(nb_iters),
      useRansac(use_ransac),
      nbSamples(nb_samples),
      filterIter(filter_iter),
      filterAlpha(filter_alpha),
      filterBeta(filter_beta),
      filterThresh(filter_thresh)
{
}

void TPS_RGBD::initMat(const cv::cuda::GpuMat& img)
{
    indexMat.create(img.size(), CV_32SC1);
    boundaryMat.create(img.size(), CV_32SC1);
    segmentedMat.create(img.size(), CV_8UC3);
    dispMat.create(img.size(), CV_32FC1);
    inliersMat.create(img.size(), CV_8UC1);

    displayImgGpu.create(img.size(), CV_32FC1);

    imgRGBA.create(img.size(), CV_8UC4);

    texRGBA = new Texture<uchar4>(imgRGBA);
    texIndex = new Texture<int>(indexMat);
    texBound = new Texture<int>(boundaryMat);
    texDisp = new Texture<float>(dispMat);
    texInliers = new Texture<unsigned char>(inliersMat);
}


void TPS_RGBD::compute(const cv::cuda::GpuMat& img,
                       const cv::cuda::GpuMat& depth)
{
    if(img.size()!= imgRGBA.size())
    {
        initMat(img);

        roi.x=0;
        roi.y=0;
        roi.width = img.cols;
        roi.height = img.rows;

        gridSizeX = (roi.width + cellSize - 1)/ cellSize;
        gridSizeY = (roi.height + cellSize - 1)/ cellSize;

        nbSuperpixels = gridSizeX*gridSizeY;

        superpixels.resize(nbSuperpixels);
        coeffs.resize(nbSuperpixels);
        samples.resize(nbSuperpixels*nbSamples);
        randStates.resize(nbSuperpixels*nbSamples);

        initRandStates_kernel<<<nbSuperpixels, nbSamples>>>(thrust::raw_pointer_cast(&randStates[0]));
    }

    cudaMemset(thrust::raw_pointer_cast(&superpixels[0]), 0, nbSuperpixels*sizeof(SuperpixelRGBD));

    cudaMemset(thrust::raw_pointer_cast(&coeffs[0]), 0, nbSuperpixels*sizeof(SuperpixelRGBDCoeffs));


    dim3 dimBlock(16, 16);
    dim3 dimGrid((roi.width + dimBlock.x - 1) / dimBlock.x,
                 (roi.height + dimBlock.y - 1) / dimBlock.y);

    //cv::cuda::cvtColor(img, imgRGBA, COLOR_BGR2HSV, 4);
    cv::cuda::cvtColor(img, imgRGBA, COLOR_BGR2BGRA, 4);
    //cv::cuda::cvtColor(img, imgRGBA, COLOR_BGR2Luv, 4);

    dim3 dimGridFull((dispMat.cols + dimBlock.x - 1) / dimBlock.x,
                     (dispMat.rows + dimBlock.y - 1) / dimBlock.y);

    if(depth.type()==CV_32FC1)
    {
        depth2disp32F_kernel<<<dimGridFull, dimBlock>>>((float*) dispMat.data,
                                                        (const float*) depth.data,
                                                        dispMat.cols,
                                                        dispMat.rows,
                                                        dispMat.step/sizeof(float),
                                                        depth.step/sizeof(float));
//        cudaDeviceSynchronize();
//        CudaCheckError();
    }
    else if(depth.type()==CV_16UC1)
    {
        depth2disp16U_kernel<<<dimGridFull, dimBlock>>>((float*) dispMat.data,
                                                        (const uint16_t*) depth.data,
                                                        dispMat.cols,
                                                        dispMat.rows,
                                                        dispMat.step/sizeof(float),
                                                        depth.step/sizeof(uint16_t));
//        cudaDeviceSynchronize();
//        CudaCheckError();
    }

    initSuperpixelsRGBD_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&coeffs[0]),
                                                      texRGBA->getTextureObject(),
                                                      (int*) indexMat.data,
                                                      (int*) boundaryMat.data,
                                                      cellSize,
                                                      gridSizeX,
                                                      gridSizeY,
                                                      roi,
                                                      indexMat.step/sizeof(int),
                                                      boundaryMat.step/sizeof(int));
//    cudaDeviceSynchronize();
//    CudaCheckError();

    mergeTPSRGBCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                     thrust::raw_pointer_cast(&coeffs[0]),
                                                                                     nbSuperpixels);
//    cudaDeviceSynchronize();
//    CudaCheckError();

    dim3 dimGrid2((roi.width/2 + dimBlock.x - 1) / dimBlock.x,
                  (roi.height/2 + dimBlock.y - 1) / dimBlock.y);

    // RGB only
    for(int k=0; k<nbIters/2; k++)
    {
        updateTPSRGB_kernel<0,0,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                               thrust::raw_pointer_cast(&coeffs[0]),
                                                               texRGBA->getTextureObject(),
                                                               (int*) indexMat.data,
                                                               (int*) boundaryMat.data,
                                                               lambdaPos,
                                                               lambdaBound,
                                                               lambdaSize,
                                                               cellSize*cellSize/4.f,
                                                               roi,
                                                               nbSuperpixels,
                                                               indexMat.step/sizeof(int),
                                                               boundaryMat.step/sizeof(int));
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                         thrust::raw_pointer_cast(&coeffs[0]),
                                                                                         nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGB_kernel<1,1,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                               thrust::raw_pointer_cast(&coeffs[0]),
                                                               texRGBA->getTextureObject(),
                                                               (int*) indexMat.data,
                                                               (int*) boundaryMat.data,
                                                               lambdaPos,
                                                               lambdaBound,
                                                               lambdaSize,
                                                               cellSize*cellSize/4.f,
                                                               roi,
                                                               nbSuperpixels,
                                                               indexMat.step/sizeof(int),
                                                               boundaryMat.step/sizeof(int));
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                         thrust::raw_pointer_cast(&coeffs[0]),
                                                                                         nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGB_kernel<0,1,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                               thrust::raw_pointer_cast(&coeffs[0]),
                                                               texRGBA->getTextureObject(),
                                                               (int*) indexMat.data,
                                                               (int*) boundaryMat.data,
                                                               lambdaPos,
                                                               lambdaBound,
                                                               lambdaSize,
                                                               cellSize*cellSize/4.f,
                                                               roi,
                                                               nbSuperpixels,
                                                               indexMat.step/sizeof(int),
                                                               boundaryMat.step/sizeof(int));
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                         thrust::raw_pointer_cast(&coeffs[0]),
                                                                                         nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGB_kernel<1,0,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                               thrust::raw_pointer_cast(&coeffs[0]),
                                                               texRGBA->getTextureObject(),
                                                               (int*) indexMat.data,
                                                               (int*) boundaryMat.data,
                                                               lambdaPos,
                                                               lambdaBound,
                                                               lambdaSize,
                                                               cellSize*cellSize/4.f,
                                                               roi,
                                                               nbSuperpixels,
                                                               indexMat.step/sizeof(int),
                                                               boundaryMat.step/sizeof(int));
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                         thrust::raw_pointer_cast(&coeffs[0]),
                                                                                         nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();
    }

    if(useRansac)
    {
        /* Init planes with RANSAC */
        initSamples_kernel<<<nbSuperpixels, nbSamples>>>(thrust::raw_pointer_cast(&samples[0]),
                                                         thrust::raw_pointer_cast(&randStates[0]),
                                                         thrust::raw_pointer_cast(&superpixels[0]),
                                                         texIndex->getTextureObject(),
                                                         texDisp->getTextureObject(),
                                                         10,
                                                         cellSize/2.f,
                                                         roi);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        evalSamples_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&samples[0]),
                                                  texIndex->getTextureObject(),
                                                  texDisp->getTextureObject(),
                                                  threshDisp,
                                                  roi,
                                                  nbSamples);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        selectSamples_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                     thrust::raw_pointer_cast(&coeffs[0]),
                                                                                     thrust::raw_pointer_cast(&samples[0]),
                                                                                     nbSuperpixels,
                                                                                     nbSamples);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        initDispCoeffsRansacRGBD_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&coeffs[0]),
                                                               inliersMat.data,
                                                               thrust::raw_pointer_cast(&superpixels[0]),
                                                               texDisp->getTextureObject(),
                                                               texIndex->getTextureObject(),
                                                               threshDisp,
                                                               roi,
                                                               inliersMat.step);
//        cudaDeviceSynchronize();
//        CudaCheckError();
        /* end RANSAC */

    }else{
        initDispCoeffsRGBD_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&coeffs[0]),
                                                         inliersMat.data,
                                                         texDisp->getTextureObject(),
                                                         texIndex->getTextureObject(),
                                                         roi,
                                                         inliersMat.step);
//        cudaDeviceSynchronize();
//        CudaCheckError();
    }

    mergeTPSRGBDCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                      thrust::raw_pointer_cast(&coeffs[0]),
                                                                                      nbSuperpixels);
//    cudaDeviceSynchronize();
//    CudaCheckError();


    for(int k=nbIters/2; k<nbIters; k++)
    {
        updateTPSRGBD_kernel<0,0,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                thrust::raw_pointer_cast(&coeffs[0]),
                                                                texRGBA->getTextureObject(),
                                                                texDisp->getTextureObject(),
                                                                (int*) indexMat.data,
                                                                (int*) boundaryMat.data,
                                                                (unsigned char*) inliersMat.data,
                                                                lambdaPos,
                                                                lambdaBound,
                                                                lambdaSize,
                                                                lambdaDisp,
                                                                threshDisp,
                                                                cellSize*cellSize/4.f,
                                                                roi,
                                                                nbSuperpixels,
                                                                indexMat.step/sizeof(int),
                                                                boundaryMat.step/sizeof(int),
                                                                inliersMat.step);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBDCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                          thrust::raw_pointer_cast(&coeffs[0]),
                                                                                          nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGBD_kernel<1,1,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                thrust::raw_pointer_cast(&coeffs[0]),
                                                                texRGBA->getTextureObject(),
                                                                texDisp->getTextureObject(),
                                                                (int*) indexMat.data,
                                                                (int*) boundaryMat.data,
                                                                (unsigned char*) inliersMat.data,
                                                                lambdaPos,
                                                                lambdaBound,
                                                                lambdaSize,
                                                                lambdaDisp,
                                                                threshDisp,
                                                                cellSize*cellSize/4.f,
                                                                roi,
                                                                nbSuperpixels,
                                                                indexMat.step/sizeof(int),
                                                                boundaryMat.step/sizeof(int),
                                                                inliersMat.step);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBDCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                          thrust::raw_pointer_cast(&coeffs[0]),
                                                                                           nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGBD_kernel<0,1,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                thrust::raw_pointer_cast(&coeffs[0]),
                                                                texRGBA->getTextureObject(),
                                                                texDisp->getTextureObject(),
                                                                (int*) indexMat.data,
                                                                (int*) boundaryMat.data,
                                                                (unsigned char*) inliersMat.data,
                                                                lambdaPos,
                                                                lambdaBound,
                                                                lambdaSize,
                                                                lambdaDisp,
                                                                threshDisp,
                                                                cellSize*cellSize/4.f,
                                                                roi,
                                                                nbSuperpixels,
                                                                indexMat.step/sizeof(int),
                                                                boundaryMat.step/sizeof(int),
                                                                inliersMat.step);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        mergeTPSRGBDCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                          thrust::raw_pointer_cast(&coeffs[0]),
                                                                                          nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

        updateTPSRGBD_kernel<1,0,16,16><<<dimGrid2, dimBlock>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                thrust::raw_pointer_cast(&coeffs[0]),
                                                                texRGBA->getTextureObject(),
                                                                texDisp->getTextureObject(),
                                                                (int*) indexMat.data,
                                                                (int*) boundaryMat.data,
                                                                (unsigned char*) inliersMat.data,
                                                                lambdaPos,
                                                                lambdaBound,
                                                                lambdaSize,
                                                                lambdaDisp,
                                                                threshDisp,
                                                                cellSize*cellSize/4.f,
                                                                roi,
                                                                nbSuperpixels,
                                                                indexMat.step/sizeof(int),
                                                                boundaryMat.step/sizeof(int),
                                                                inliersMat.step);

        mergeTPSRGBDCoeffs_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                          thrust::raw_pointer_cast(&coeffs[0]),
                                                                                          nbSuperpixels);
//        cudaDeviceSynchronize();
//        CudaCheckError();

    }

    cudaDeviceSynchronize();

    //   renderSegmentedImage_kernel<<<dimGrid, dimBlock>>>(
    //     (unsigned char*) segmentedMat.data,
    //     thrust::raw_pointer_cast(&superPixels[0]),
    //     (int*) indexMat.data,
    //     roi.width,
    //     roi.height,
    //     segmentedMat.step,
    //     indexMat.step/sizeof(int));

    //   renderBoundaryImage_kernel<<<dimGrid, dimBlock>>>(
    //     (unsigned char*) segmentedMat.data,
    //     texRGBA->getTextureObject(),
    //     (int*) indexMat.data,
    //     roi.width,
    //     roi.height,
    //     segmentedMat.step,
    //     indexMat.step/sizeof(int));


    /*
  displayImgGpu.download(displayImg);
  imshow("segmented_depth", displayImg);

  inliersMat.download(displayImg2);
  imshow("inliers", displayImg2);
  */
}

void TPS_RGBD::filter()
{
    filterData.resize(superpixels.size());
    initFilter_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&filterData[0]),
                                                                              thrust::raw_pointer_cast(&superpixels[0]),
                                                                              nbSuperpixels);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((gridSizeX + dimBlock.x - 1) / dimBlock.x,
                 (gridSizeY + dimBlock.y - 1) / dimBlock.y);
    for(int k=0; k<filterIter; k++)
    {
        iterateFilter_kernel<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(&filterData[0]),
                                                    filterAlpha,
                                                    filterBeta,
                                                    filterThresh,
                                                    gridSizeX,
                                                    gridSizeY);
    }


    finishFilter_kernel<<<(nbSuperpixels+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(thrust::raw_pointer_cast(&superpixels[0]),
                                                                                thrust::raw_pointer_cast(&filterData[0]),
                                                                                nbSuperpixels);

}

void TPS_RGBD::computeDepthImage(cv::cuda::GpuMat& depthMat)
{
    depthMat.create(dispMat.size(), CV_32FC1);

    dim3 dimBlock(16, 16);

    dim3 dimGridFull((dispMat.cols + dimBlock.x - 1) / dimBlock.x,
                     (dispMat.rows + dimBlock.y - 1) / dimBlock.y);

    renderDepthImage_kernel<<<dimGridFull, dimBlock>>>((float*) depthMat.data,
                                                       thrust::raw_pointer_cast(&superpixels[0]),
                                                       texIndex->getTextureObject(),
                                                       texInliers->getTextureObject(),
                                                       roi,
                                                       1.f,
                                                       depthMat.cols,
                                                       depthMat.rows,
                                                       depthMat.step/sizeof(float));
}

void TPS_RGBD::computePreviewImage(cv::cuda::GpuMat& previewImg)
{
    previewImg.create(roi.size(), CV_8UC3);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((roi.width + dimBlock.x - 1) / dimBlock.x,
                 (roi.height + dimBlock.y - 1) / dimBlock.y);

    renderBoundaryImage_kernel<<<dimGrid, dimBlock>>>((unsigned char*) previewImg.data,
                                                        texRGBA->getTextureObject(),
                                                        (int*) indexMat.data,
                                                        roi,
                                                        previewImg.step,
                                                        indexMat.step/sizeof(int));
}


} // namespace supersurfel_fusion
