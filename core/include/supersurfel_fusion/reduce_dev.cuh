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


#ifndef REDUCE_DEV_CUH
#define REDUCE_DEV_CUH


template <typename T>
__device__  void warpReduce(T* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  __syncwarp();
  sdata[tid] += sdata[tid + 16];
  __syncwarp();
  sdata[tid] += sdata[tid + 8];
  __syncwarp();
  sdata[tid] += sdata[tid + 4];
  __syncwarp();
  sdata[tid] += sdata[tid + 2];
  __syncwarp();
  sdata[tid] += sdata[tid + 1];
  __syncwarp();
}

template <int N, typename T>
  __device__ __forceinline__ void reduce(T* shmem, int tid)
{
  if(N >= 2048)
  {
    if (tid < 1024)
      shmem[tid]+=shmem[tid+1024];
    __syncthreads();
  }
  if(N >= 1024)
  {
    if (tid < 512)
      shmem[tid]+=shmem[tid+512];
    __syncthreads();
  }
  if(N >= 512)
  {
    if (tid < 256)
      shmem[tid]+=shmem[tid+256];
    __syncthreads();
  }
  if(N >= 256)
  {
    if (tid < 128)
      shmem[tid]+=shmem[tid+128];
    __syncthreads();
  }
  if(N >= 128)
  {
    if (tid < 64)
      shmem[tid]+=shmem[tid+64];
    __syncthreads();
  }
  if (tid < 32) warpReduce(shmem, tid);
}

template <typename T>
__device__ __forceinline__ void reduce(T* shmem, int tid, unsigned int N)
{
  for (unsigned int s=N/2; s>32; s>>=1) {
  if (tid < s)
    shmem[tid] += shmem[tid + s];
    __syncthreads();
  }
  if (tid < 32) warpReduce(shmem, tid);
}


//template <int N, typename T>
//  __device__ __forceinline__ void reduce(T* shmem, int tid)
//{
//  if(N >= 2048)
//  {
//    if (tid < 1024)
//      shmem[tid]+=shmem[tid+1024];
//    __syncthreads();
//  }
//  if(N >= 1024)
//  {
//    if (tid < 512)
//      shmem[tid]+=shmem[tid+512];
//    __syncthreads();
//  }
//  if(N >= 512)
//  {
//    if (tid < 256)
//      shmem[tid]+=shmem[tid+256];
//    __syncthreads();
//  }
//  if(N >= 256)
//  {
//    if (tid < 128)
//      shmem[tid]+=shmem[tid+128];
//    __syncthreads();
//  }
//  if(N >= 128)
//  {
//    if (tid < 64)
//      shmem[tid]+=shmem[tid+64];
//    __syncthreads();
//  }
//  if(N >= 64)
//  {
//    if (tid < 32)
//      shmem[tid]+=shmem[tid+32];
//    __syncthreads();
//  }
  
//  if (tid < 16)
//  {
//    shmem[tid]+=shmem[tid+16];
//    //__syncthreads();
//    shmem[tid]+=shmem[tid+8];
//    //__syncthreads();
//    shmem[tid]+=shmem[tid+4];
//    //__syncthreads();
//    shmem[tid]+=shmem[tid+2];
//    //__syncthreads();
//    shmem[tid]+=shmem[tid+1];
//    //__syncthreads();
//  }//else{
//    //__syncthreads();
//    //__syncthreads();
//    //__syncthreads();
//    //__syncthreads();
//    //__syncthreads();
//  //}
//}

#endif // REDUCE_DEV_CUH
