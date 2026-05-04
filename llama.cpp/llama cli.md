#llamacpp #DeepDive

# 1. Overal architecture

The `llama-cli` application has 2 threads:
+ main thread which act as UI thread in desktop/mobile application. It receive user commands, prompts; offload tasks to worker threads; wait for results; present results to users
+ worker thread(s) which receive tasks from main thread; do the works to generate results; send results back to the main thread

It's the classic producer