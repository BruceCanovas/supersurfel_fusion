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


#ifndef CUDA_UTILS_DEV_CUH
#define CUDA_UTILS_DEV_CUH

#if CUDA_VERSION >= 9000
#include <cooperative_groups.h>
#endif

#if CUDA_VERSION>=9000
inline __device__ int atomicAggInc(int *ctr) {
    cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
    int warp_res;
    if(g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}

inline __device__ int atomicAggDec(int *ctr) {
    cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
    int warp_res;
    if(g.thread_rank() == 0)
        warp_res = atomicSub(ctr, g.size());
    return g.shfl(warp_res, 0) - g.thread_rank();
}

inline __device__ unsigned int atomicAggInc(unsigned int *ctr) {
    cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
    unsigned int warp_res;
    if(g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}

inline __device__ unsigned int atomicAggDec(unsigned int *ctr) {
    cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
    unsigned int warp_res;
    if(g.thread_rank() == 0)
        warp_res = atomicSub(ctr, g.size());
    return g.shfl(warp_res, 0) - g.thread_rank();
}
#else
inline __device__ int atomicAggInc(int *ctr) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & __lanemask_lt());
    int warp_res;
    if(rank == 0)
        warp_res = atomicAdd(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
}

inline __device__ int atomicAggDec(int *ctr) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & __lanemask_lt());
    int warp_res;
    if(rank == 0)
        warp_res = atomicSub(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res - rank;
}

inline __device__ unsigned int atomicAggInc(unsigned int *ctr) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & __lanemask_lt());
    unsigned int warp_res;
    if(rank == 0)
        warp_res = atomicAdd(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res + rank;
}

inline __device__ unsigned int atomicAggDec(unsigned int *ctr) {
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;
    int change = __popc(active);
    unsigned int rank = __popc(active & __lanemask_lt());
    unsigned int warp_res;
    if(rank == 0)
        warp_res = atomicSub(ctr, change);
    warp_res = __shfl_sync(active, warp_res, leader);
    return warp_res - rank;
}
#endif //CUDA_VERSION>=9000

#endif // __CUDA_UTILS_DEV_CUH__
