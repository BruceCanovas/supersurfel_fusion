// Source taken from Mihail Ivakhnenko implementation, from Nathan Bell
// and Jared Hoberock example
//
// This class allow to intercept calls to get_temporary_buffer
// and return_temporary_buffer to control how Thrust allocates temporary storage
// during algorithms such as thrust::reduce. The idea will be to create a simple
// cache of allocations to search when temporary storage is requested. If a hit
// is found in the cache, we quickly return the cached allocation instead of
// resorting to the more expensive thrust::cuda::malloc.
//
// Note: this implementation CachedAllocator is not thread-safe. If multiple
// (host) threads use the same CachedAllocator then they should gain exclusive
// access to the allocator before accessing its methods.


#ifndef __CACHED_ALLOCATOR_H__
#define __CACHED_ALLOCATOR_H__

#include <cuda_runtime.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <iostream>
#include <map>


namespace supersurfel_fusion
{

// CachedAllocator: a simple allocator for caching allocation requests
class CachedAllocator
{

public:
    // just allocate bytes
    typedef char value_type;

    CachedAllocator();

    ~CachedAllocator();

    char* allocate(std::ptrdiff_t num_bytes);

    void deallocate(char* ptr, size_t n);

private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all();

        //std::ptrdiff_t allocated_mem;
};

} // namespace supersurfel_fusion

#endif // __CACHED_ALLOCATOR_H__
