#Pytorch  #MemoryManagement #CUDA #DataStructure 

# 1. Overview of CUDA allocator

- The Pytorch project implement its own CUDA memory manager

## 1.1. CUDA allocator initialization

```C++
struct BackendStaticInitializer {
  // Parses env for backend at load time, duplicating some logic from
  // CUDAAllocatorConfig. CUDAAllocatorConfig double-checks it later (at
  // runtime). Defers verbose exceptions and error checks, including Cuda
  // version checks, to CUDAAllocatorConfig's runtime doublecheck. If this
  // works, maybe we should move all of CUDAAllocatorConfig here?
  CUDAAllocator* parseEnvForBackend() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    ...
    if (kv[0] == "backend") {
        if (kv[1] == "cudaMallocAsync")
            return CudaMallocAsync::allocator();
        if (kv[1] == "native")
            return &Native::allocator;
    }
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

std::atomic<CUDAAllocator*> allocator;
BackendStaticInitializer backend_static_initializer;
```

- At runtime, environment variable is checked to decide which backend allocator is used. Currently (v2.5.0) there are 2 choices: `CudaMallocAsync` and `Native`. In which the `Native` allocator is the default.
- As the name suggests, the allocator caches allocated memory blocks for future uses.

## 1.2. Implementation of CUDA native allocator

### 1.2.1. NativeCachingAllocator

```C++
class NativeCachingAllocator : public CUDAAllocator {
  std::array<ska::flat_hash_map<void*, Block*>, kNumMutexShard>
      allocated_blocks;
 public:
  void malloc(
      void** devPtr,
      c10::DeviceIndex device,
      size_t size,
      cudaStream_t stream) {
    ...
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    ... 
  }

  void free(void* ptr) {
    ...
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    device_allocator[block->device]->free(block);
    ...
  }
};
```

- The allocator keep track of all memory blocks in-use in all of the devices.
- The allocator expose the `malloc` and `free` APIs for end-users which works on raw pointers ==> The class keep track of the allocated pointers and its corresponding internal data structures.
- The internal data structures `Block` is extracted inside `malloc` and `free`, and then forwarded to `DeviceCachingAllocator`'s `malloc` and `free`. 
- The class `DeviceCachingAllocator` do the heavy lifting of memory management.

### 1.2.2. DeviceCachingAllocator

- Allocations are associated with a stream. Once freed, blocks can be re-allocated on the same stream, but not on any other stream.
- The allocator attempts to find the smallest cached block that will fit the requested size. If the block is larger than the requested size, it may be split. If no block is found, the allocator will delegate to `cudaMalloc`.
- If the cudaMalloc fails, the allocator will attempt to free one cached block of sufficient size that is not split and retry the allocation. If this also fails, the allocator will attempt to free all cached blocks that are not split and retry the allocation.
- Large (>1MB) and small allocations are stored in separate pools. Small requests are packed into 2MB buffers. Large requests will use the smallest available free block or allocate a new block using cudaMalloc.
- To reduce fragmentation, requests between 1MB and 10MB will allocate and split a 20MB block, if no free block of sufficient size is available.
- To further reduce fragmentation, blocks >= max_split_size are not allowed to be split. These oversize cached blocks will still satisfy requests within 1MB of the oversize cached block size.

### 1.2.3. Block

- class `Block` is the data structure which hold the actual allocated memory
- When allocator classes call allocate, the call will eventually forwarded to class `Block` if no available caches are found.
- class `Block` is not a dumb forwarder to `cudaMalloc`. It take advantage of CUDA's driver API regarding virtual memory management.
- When constructed, the `Block` instance allocate very large chunk of virtual memory without physical memory backing ==> The `Block` instance is contiguous and rarely need to be re-allocated.
- However, the large chunk of virtual memory does not hog actual memory. In fact, physical memory is acquired on-demands and map to the virtual memory space.
- The `Block`'s managing of memory is transparent to the allocators.