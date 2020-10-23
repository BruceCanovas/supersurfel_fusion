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


#include <supersurfel_fusion/cached_allocator.hpp>


namespace supersurfel_fusion
{

inline int
pow2roundup (int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

CachedAllocator::CachedAllocator()
//  :allocated_mem(0)
{
}

CachedAllocator::~CachedAllocator()
{
  // free all allocations when CachedAllocator goes out of scope
  free_all();
}

char* CachedAllocator::allocate(std::ptrdiff_t num_bytes)
{
  char* result = 0;

  std::ptrdiff_t mem_size = pow2roundup(num_bytes);

  // search the cache for a free block
  free_blocks_type::iterator free_block = free_blocks.find(mem_size);

  if (free_block != free_blocks.end())
  {
        //std::cout << "CachedAllocator::allocator(): found a hit" << std::endl;

        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
  }
  else
  {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
        {
          //std::cout << "CachedAllocator::allocator(): no free block found; calling cuda::malloc (" <<  mem_size << ")"<< std::endl;

          // allocate memory and convert cuda::pointer to raw pointer
          result = thrust::cuda::malloc<char>(mem_size).get();

      //std::cout << (long long int)result << std::endl;

      /*if(cudaMalloc(&result, mem_size)==cudaSuccess)
      {
        std::cout << "allocated "<< mem_size << " bytes of device memory" << std::endl;
      }else{
        std::cout << "fail to allocate " << mem_size << " bytes of device memory" << std::endl;
      }*/


          //allocated_mem += mem_size;
          //std::cout << "CachedAllocator::allocator(): memory blocks : " << allocated_blocks.size() << std::endl;
        }
        catch(std::runtime_error &e)
        {
          throw;
        }
  }

  // insert the allocated pointer into the allocated_blocks map
  allocated_blocks.insert(std::make_pair(result, mem_size));

  //std::cout << "Allocated block size : " << num_bytes << std::endl;
  //std::cout << "Allocated blocks : " << allocated_blocks.size() << std::endl;

  return result;
}

void CachedAllocator::deallocate(char* ptr, size_t n)
{
  // erase the allocated block from the allocated blocks map
  allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
  std::ptrdiff_t num_bytes = iter->second;
  allocated_blocks.erase(iter);

  // insert the block into the free blocks map
  free_blocks.insert(std::make_pair(num_bytes, ptr));
}

void CachedAllocator::free_all()
{
  std::cout << "CachedAllocator::free_all(): cleaning up after ourselves..." << std::endl;

  // deallocate all outstanding blocks in both lists
  for (free_blocks_type::iterator i = free_blocks.begin();
           i != free_blocks.end(); i++)
  {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
  }

  for (allocated_blocks_type::iterator i = allocated_blocks.begin();
           i != allocated_blocks.end(); i++)
  {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
  }

  //allocated_mem = 0;
}

} // namespace supersurfel_fusion
