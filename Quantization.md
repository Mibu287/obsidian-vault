
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
 
  /*! \brief data field */
  Entry *data;
  /*! \brief number of elements in the summary */
  size_t size;
```

```c++
template<typename DType, typename RType, class TSummary>
class QuantileSketchTemplate {

};
```