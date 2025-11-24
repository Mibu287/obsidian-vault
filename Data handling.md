#xgboost #DataStructure #DeepDive #MachineLearning

# 1. Data structure

xgboost library manage data in form of matrix (each row is an observation, each column is a feature). Observation is stored in form of CSR matrix (compressed sparse row).

```c++
namespace xgboost::data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
  MetaInfo info_;
  // Primary storage type
  std::shared_ptr<SparsePage> sparse_page_ = std::make_shared<SparsePage>();
  std::shared_ptr<CSCPage> column_page_{nullptr};
  std::shared_ptr<SortedCSCPage> sorted_column_page_{nullptr};
  std::shared_ptr<EllpackPage> ellpack_page_{nullptr};
  std::shared_ptr<GHistIndexMatrix> gradient_index_{nullptr};
  BatchParam batch_param_;
};

class SparsePage {
 public:
  // Offset for each row.
  HostDeviceVector<bst_idx_t> offset;
  /*! \brief the data of the segments */
  HostDeviceVector<Entry> data;

  size_t base_rowid {0};
};

struct Entry {
  /*! \brief feature index */
  bst_feature_t index;
  /*! \brief feature value */
  bst_float fvalue;
};
```

As the code above showed, the class which implement data structure for a XGBoost dataset is `SimpleDMatrix` which mainly consisted of `SparsePage`. In which, `SparsePage` contains vector of entries and corresponding offsets.

E.g:

entries: | f1, v1 | f2, v2 | f3, v3| f2, v2 | f4, v4|
offset. : 0 | 3 | 5

==> The first row consists of entries in range \[0, 3) which are 3 features f1, f2, f3. The second row consists of entries in range [3]