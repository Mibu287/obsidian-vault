
#xgboost #MachineLearning #DeepDive 


# 1. 2D Space

```c++
// Split 2d space to balanced blocks
// Implementation of the class is inspired by tbb::blocked_range2d
// However, TBB provides only (n x m) 2d range (matrix) separated by blocks. Example:
// [ 1,2,3 ]
// [ 4,5,6 ]
// [ 7,8,9 ]
// But the class is able to work with different sizes in each 'row'. Example:
// [ 1,2 ]
// [ 3,4,5,6 ]
// [ 7,8,9]
// If grain_size is 2: It produces following blocks:
// [1,2], [3,4], [5,6], [7,8], [9]
// The class helps to process data in several tree nodes (non-balanced usually) in parallel
// Using nested parallelism (by nodes and by data in each node)
// it helps to improve CPU resources utilization
class BlockedSpace2d {
 public:
  // Example of space:
  // [ 1,2 ]
  // [ 3,4,5,6 ]
  // [ 7,8,9]
  // BlockedSpace2d will create following blocks (tasks) if grain_size=2:
  // 1-block: first_dimension = 0, range of indexes in a 'row' = [0,2) (includes [1,2] values)
  // 2-block: first_dimension = 1, range of indexes in a 'row' = [0,2) (includes [3,4] values)
  // 3-block: first_dimension = 1, range of indexes in a 'row' = [2,4) (includes [5,6] values)
  // 4-block: first_dimension = 2, range of indexes in a 'row' = [0,2) (includes [7,8] values)
  // 5-block: first_dimension = 2, range of indexes in a 'row' = [2,3) (includes [9] values)
  std::vector<Range1d> ranges_;
  std::vector<std::size_t> first_dimension_;
};
```

The struct is used to split 2D space into roughly equal chunks. This is used to split works for multi-threaded programs.

E.g: Find best splits for multiple leaves, each leaf need to be considered multiple features.

==> 2D space consists of 1st dimension is candidate leaves, 2nd dimension is features. Each candidate leaves may has different number of features.


# 2. Dispatch jobs on thread pool

```c++
template <typename Func>
void ParallelFor2d(const BlockedSpace2d& space, int n_threads, Func&& func) {
  static_assert(std::is_void_v<std::invoke_result_t<Func, std::size_t, Range1d>>);
  std::size_t n_blocks_in_space = space.Size();
  CHECK_GE(n_threads, 1);

  dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
  {
    exc.Run([&]() {
      std::size_t tid = omp_get_thread_num();
      std::size_t chunck_size = n_blocks_in_space / n_threads + !!(n_blocks_in_space % n_threads);

      std::size_t begin = chunck_size * tid;
      std::size_t end = std::min(begin + chunck_size, n_blocks_in_space);
      for (auto i = begin; i < end; i++) {
        func(space.GetFirstDimension(i), space.GetRange(i));
      }
    });
  }
  exc.Rethrow();
}
```

xgboost library use OpenMP for job dispatching. When jobs are sent to thread pool, each job determine its own chunk of works based on its thread number and total number of threads. Then it do work on its own chunks.

Proof that 