
#xgboost #DataStructure #MachineLearning #DeepDive 


# 1. Quantile

xgboost library use weighted quantile sketch algorithm (detailed as in the paper) to separate the dataset in roughly equal weighted bins (number of bins are set as a hyper-parameter). Split finding algorithm works on the bins (not original dataset) to speed up the training process.

```c++
template<typename DType, typename RType>
struct WQSummary {
  /*! \brief an entry in the sketch summary */
  struct Entry {
    /*! \brief minimum rank */
    RType rmin{};
    /*! \brief maximum rank */
    RType rmax{};
    /*! \brief maximum weight */
    RType wmin{};
    /*! \brief the value of data */
    DType value{};
  };
  
  struct Queue {
    // entry in the queue
    struct QEntry {
      // value of the instance
      DType value;
      // weight of instance
      RType weight;
    };
    // the input queue
    std::vector<QEntry> queue;
    // end of the queue
    size_t qtail;
  };
 
  /*! \brief data field */
  Entry *data;
  /*! \brief number of elements in the summary */
  size_t size;
```

```c++
template<typename DType, typename RType, class TSummary>
class QuantileSketchTemplate {
  typename Summary::Queue inqueue;
  // number of levels
  size_t nlevel;
  // size of summary in each level
  size_t limit_size;
  // the level of each summaries
  std::vector<Summary> level;
  // content of the summary
  std::vector<Entry> data;
  // temporal summary, used for temp-merge
  SummaryContainer temp;
};

template<typename DType, typename RType = unsigned>
class WQuantileSketch :
      public QuantileSketchTemplate<DType, RType, WQSummary<DType, RType> > { };
```

The histogram of input data is typically built only once at initialization of the learner object. Subsequent iteration can reuse the histogram ==> Speed up the process massively.

The algorithm is roughly as following:

1. Data is ingested into learner object. As data is being ingested, it is stored in queue.
2. When the queue is full, it is processed to move to `WQSummary` at level 0.
3. Check if the summary at level 0 is full. If not, move on to the next iteration. If yes, move up 1 level, combined with existing summary the the level. Then, prune the combined summary to fit budgeted size.
4. Recursively repeat step 3.

E.g:
Input data has 256 * 2^N observations, expected bin count is 256.
h is number of levels. Since each level add error to final cut ==> each level is required to maintain 1 / (256 * h) error budget (refer to the xgboost paper for more info) ==> budget space is (256 * h).
Find minimum h satisfy 2^h * (256 * h) > 256 * 2^N
Space requirement = O(h^2)

==> Space requirement grow roughly at magnitude of logarithm of input size (slow growth)

When the whole dataset is ingested, the quantile sketch is used to make bins of roughly equal weight. Then the sketch is dropped.


# 2. Histogram

