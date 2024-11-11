#DeepLearning #Pytorch #DeepDive #DataStructure 

# 1. What is a Tensor?

- a Tensor is a multi-dimensional array of single data type
  e.g:

```text
  Tensor([
    [1, 2],
    [2, 3]
  ]), dtype=float32
```

- What can a Tensor do?
  - Arithmetic operations with other tensors / scalers

# 2. Tensor in Pytorch

- Python object, C++ implementation (ATen library)

- Interaction with other Python array library (numpy)

```python
np_array = np.ones((2, 2))
torch_array = torch.tensor(np_array) # New copy is made
torch_array = torch.from_numpy(np_array) # A ref to np array is made
```

- Implementation of ```from_numpy``` method:

```cpp
at::Tensor tensor_from_numpy(PyObject* obj, ...) {
    auto array = (PyArrayObject*)obj;
    int ndim = ...,
    auto sizes = ...;
    auto strides = ...;
    void* data_ptr = PyArray_DATA(array);
    Py_INCREF(obj);
    return at::lift_fresh(at::from_blob(
        data_ptr, sizes, strides, ...
    ));
};
```

- Tensor storage
  - Abstraction to separate raw data and its interpretations.
      -e.g: 2 tensors share the same storage but has different views
  - Memory allocators:
    - CPU and/or GPU
    - PyTorch use the abstract class Allocator
  - Each Tensor has ```Storage *storage``` field which is pointer to Storage object
    - The Storage object is a pointer to the raw data and allocator object
- Tensor trinity
  - Device: Which device the tensor reside
  - Layout: How data is stored in the tensor (strided, sparse, mkldnn, ...)
  - dtype: Type of the data inside the tensor (1 dtype / tensor)
- Each tensor is a 3-tuple of the 3 properties (Device, Layout, dtype). Althought, not all tensor has repective kernels.

# 3. Exploring Pytorch's source code about Tensor

## 3.1. Tensor definition

### 3.1.1. class Tensor

```C++
// file: aten/src/ATen/core/TensorBody.h
// The source file is auto-generated from template
class TORCH_API Tensor: public TensorBase {
    // No additional data-member compared to TensorBase
    // Only method definitions
    ...
};
```

### 3.1.2. class TensorBase

```C++
// file: aten/src/ATen/core/TensorBase.h
// Tensor, being the central data structure in PyTorch, gets used and
// it's header included almost everywhere. Unfortunately this means
// every time an operator signature is updated or changed in
// native_functions.yaml, you (and every other PyTorch developer) need
// to recompile all of ATen and it's dependencies.
//
// TensorBase aims to break up these header dependencies, and improve
// incremental build times for all PyTorch developers. TensorBase
// represents a reference counted handle to TensorImpl, exactly the
// same as Tensor. However, TensorBase doesn't have code generated
// methods in it's API and thus no dependence on native_functions.yaml.
class TensorBase {
protected:
    // TensorBase is just ref-counted pointer to TensorImpl
    c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
```

### 3.1.3. struct TensorImpl

```C++
// file: c10/core/TensorImpl.h
/**
 * The low-level representation of a tensor, which contains a pointer
 * to a storage (which contains the actual data) and metadata (e.g., sizes and
 * strides) describing this particular view of the data as a tensor.
 *
 * Some basic characteristics about our in-memory representation of
 * tensors:
 *
 *  - It contains a pointer to a storage struct (Storage/StorageImpl)
 *    which contains the pointer to the actual data and records the
 *    data type and device of the view.  This allows multiple tensors
 *    to alias the same underlying data, which allows to efficiently
 *    implement differing *views* on a tensor.
 *
 *  - The tensor struct itself records view-specific metadata about
 *    the tensor, e.g., sizes, strides and offset into storage.
 *    Each view of a storage can have a different size or offset.
 *
 *  - This class is intrusively refcounted.  It is refcounted so that
 *    we can support prompt deallocation of large tensors; it is
 *    intrusively refcounted so that we can still perform reference
 *    counted operations on raw pointers, which is often more convenient
 *    when passing tensors across language boundaries.
 *
 *  - For backwards-compatibility reasons, a tensor may be in an
 *    uninitialized state.  A tensor may be uninitialized in the following
 *    two ways:
 *
 *      - A tensor may be DTYPE UNINITIALIZED.  A tensor of this
 *        form has an uninitialized dtype.  This situation most
 *        frequently arises when a user writes Tensor x(CPU).  The dtype
 *        is subsequently initialized when mutable_data<T>() is
 *        invoked for the first time.
 *
 *      - A tensor may be STORAGE UNINITIALIZED.  A tensor of this form
 *        has non-zero size, but has a storage with a null data pointer.
 *        This situation most frequently arises when a user calls
 *        Resize() or FreeMemory().  This is because Caffe2 historically
 *        does lazy allocation: allocation of data doesn't occur until
 *        mutable_data<T>() is invoked.  A tensor with zero size is
 *        always storage initialized, because no allocation is necessary
 *        in this case.
 *
 *    All combinations of these two uninitialized states are possible.
 *    Consider the following transcript in idiomatic Caffe2 API:
 *
 *      Tensor x(CPU); // x is storage-initialized, dtype-UNINITIALIZED
 *      x.Resize(4); // x is storage-UNINITIALIZED, dtype-UNINITIALIZED
 *      x.mutable_data<float>(); // x is storage-initialized, dtype-initialized
 *      x.FreeMemory(); // x is storage-UNINITIALIZED, dtype-initialized.
 *
 *    All other fields on tensor are always initialized.  In particular,
 *    size is always valid. (Historically, a tensor declared as Tensor x(CPU)
 *    also had uninitialized size, encoded as numel == -1, but we have now
 *    decided to default to zero size, resulting in numel == 0).
 *
 *    Uninitialized storages MUST be uniquely owned, to keep our model
 *    simple.  Thus, we will reject operations which could cause an
 *    uninitialized storage to become shared (or a shared storage to
 *    become uninitialized, e.g., from FreeMemory).
 *
 *    In practice, tensors which are storage-UNINITIALIZED and
 *    dtype-UNINITIALIZED are *extremely* ephemeral: essentially,
 *    after you do a Resize(), you basically always call mutable_data()
 *    immediately afterwards.  Most functions are not designed to
 *    work if given a storage-UNINITIALIZED, dtype-UNINITIALIZED tensor.
 *
 *    We intend to eliminate all uninitialized states, so that every
 *    tensor is fully initialized in all fields.  Please do not write new code
 *    that depends on these uninitialized states.
 */
struct C10_API TensorImp
  Storage storage_;
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;     std::unique_ptr<c10::ExtraMeta> extra_meta_ = nullptr;
  c10::VariableVersion version_counter_;
  impl::PyObjectSlot pyobj_slot_;
  c10::impl::SizesAndStrides sizes_and_strides_;
  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;
  caffe2::TypeMeta data_type_;
  std::optional<c10::Device> device_opt_;
  DispatchKeySet key_set_;
  ...
};
```

- Refer [[Intrusive pointer#1.1. Intrusive pointer target]] for more information about  `c10::intrusive_ptr_target`
- Refer [[TypeMeta]] for more information about `TypeMeta`
- Refer [[Tensor Storage]] for more information about `Storage`
- Refer [[Dispatcher#1.2. Dispatch Key Set]] for more information about `DispatchKeySet`