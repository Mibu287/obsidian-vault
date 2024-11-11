#Pytorch #DataStructure #DeepLearning #DeepDive #Miscellaneous 

# 1. Unique void pointer
 
 A `detail::UniqueVoidPtr` is an owning smart pointer like unique_ptr, but
 with three major differences:

1. It is specialized to void
2. It is specialized for a function pointer deleter `void(void* ctx);` i.e., the deleter doesn't take a reference to the data, just to a context pointer (erased as void*).  In fact, internally, this pointer is implemented as having an owning reference to context, and a non-owning reference to data; this is why you release_context(), not release() (the conventional API for release() wouldn't give you enough information to properly dispose of the object later.)
3. The deleter is guaranteed to be called when the unique pointer is destructed and the context is non-null; this is different from std::unique_ptr where the deleter is not called if the data pointer is null.

 Some of the methods have slightly different types than std::unique_ptr to reflect this.

```C++
using DeleterFnPtr = void (*)(void*);

class UniqueVoidPtr {
  // Lifetime tied to ctx_
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;
};
```

In Pytorch code, the data does not exist in a vacuum but in a larger context (Tensor, ...) ==> If `std::unique_ptr` is used, the deleter must know about the context pointer and act accordingly. ==In addition, the deleter is not called when data pointer is NULL which may leak the context object.
==> Pytorch author must create their own unique pointer for their own needs.
