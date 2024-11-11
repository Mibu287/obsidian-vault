#Pytorch #DeepLearning #DeepDive #DataStructure 

# 1. Definition

```C++
// file: torch/include/c10/core/Storage.h
struct C10_API Storage {
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

Refer [[Intrusive pointer#1. Intrusive pointer]] for more information about `c10::intrusive_ptr`

In essence, `Storage` is just a pointer to actual storage implementation.

**Why redirection is needed?**
- `Tensor` in Pytorch is designed as as view to actual data ==> allow many Tensor to share the same storage ==> Tensor can only hold a shared pointer to Storage.
- Another design choice, `Tensor` own the Storage ==> no need for indirection. Other data structure want to share Storage must create a `TensorView` which as the name suggest is a view of the original `Tensor`

Definition of `StorageImpl`:

```C++
  DataPtr data_ptr_;
  SymInt size_bytes_;
  bool size_bytes_is_heap_allocated_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  // All special checks in data/data_ptr calls are guarded behind this single
  // boolean. This is for performance: .data/.data_ptr calls are commonly in the
  // hot-path.
  bool has_data_ptr_check_ = false;
  // If we should throw when mutable_data_ptr() or mutable_data() is called.
  bool throw_on_mutable_data_ptr_ = false;
  // If we warn when mutable_data_ptr() or mutable_data() is called.
  bool warn_deprecated_on_mutable_data_ptr_ = false;
  Allocator* allocator_;
  impl::PyObjectSlot pyobj_slot_;
```

Definitions of `DataPtr`

```C++
// file: c10/core/Allocator.h
// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class C10_API DataPtr {
  c10::detail::UniqueVoidPtr ptr_;
  Device device_;
};
```

Refer to [[Miscellaneous#1. Unique void pointer]] for more information about `UniqueVoidPtr`
Refer to [[Device#2. Device]] for more information about `Device`