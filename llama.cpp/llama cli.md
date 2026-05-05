#llamacpp #DeepDive

# 1. Overal architecture

The `llama-cli` application has 2 threads:
+ main thread which act as UI thread in desktop/mobile application. It receive user commands, prompts; offload tasks to worker threads; wait for results; present results to users
+ worker thread(s) which receive tasks from main thread; do the works to generate results; send results back to the main thread

The main thread and worker threads communicate using 2 queues:
- `queue_tasks` to store tasks to be done
- `queue_results` to store results/partial results

Results are computed piece by piece and streamed to consumers.

# 2. Memory allocation for 