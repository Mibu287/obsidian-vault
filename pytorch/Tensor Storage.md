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

**Why a redirection is needed?**
- `Tensor` in Pytorch is designed as as view to actual data ==> allow many Tensor to share the same storage ==> Tensor can only hold a shared pointer to Storage.
- Another design choice, `Tensor` own the Storage ==> no need for indirection. Other data structure want to share Storage must create a `TensorView` which as the name suggest is a view of the original `Tensor`
