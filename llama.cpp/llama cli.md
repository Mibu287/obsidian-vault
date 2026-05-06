#llamacpp #DeepDive

# 1. Overal architecture

The `llama-cli` application has 2 threads:
+ main thread which act as UI thread in desktop/mobile application. It receive user commands, prompts; offload tasks to worker threads; wait for results; present results to users
+ worker thread(s) which receive tasks from main thread; do the works to generate results; send results back to the main thread

The main thread and worker threads communicate using 2 queues:
- `queue_tasks` to store tasks to be done
- `queue_results` to store results/partial results

Results are computed piece by piece and streamed to consumers.



```cpp
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };
```
# 2. Memory allocation

## 2.1. Input parameters

Input parameters are read from GGUF file and is `mmaped` into main memory. Each input tensor contain a pointer to a chunk of this memory arena.

## 2.2. Intermediate and output tensors

- Create compute graph from the model file(s)
- Compute total memory needed for computation by traversing compute graph in topological order. At each step, if a tensor is no longer needed, it memory requirements is freed. Free means that its memory slots is marked as available to reused by other tensors. At the end of the process, total size of the memory pool and chunk info for each tensors in the graph are calculated.
- Allocate the memory pool planned in the previous step.
- Each tensor's data pointer is filled with chunk infos in step 2.