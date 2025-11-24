#Pytorch #DataStructure #DeepDive #DeepLearning 

# 1. AutogradMetaInterface

```C++
// file: c10/core/TensorImpl.h
struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(
      bool requires_grad,
      at::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& mutable_grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual const at::Tensor& fw_grad(uint64_t level, const at::TensorBase& self)
      const = 0;
  virtual void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) = 0;
  virtual ~AutogradMetaInterface();
};
```

Each `Tensor` hold a pointer to `AutogradMetaInterface`. Other autograd functions can call this interface's methods to work on the tensor's gradient.

Important methods including: `grad` `mutable_grad`

For most tensors, the actual implementation of `AutogradMetaInterface` is `AutogradMeta`


# 2. AutogradMeta

```C++
// file: torch/csrc/autograd/variable.h
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  // This field is used to store all the forward AD gradients
  // associated with this AutogradMeta (and the Tensor it corresponds to)
  // There is a semantic 1:1 correspondence between AutogradMeta and
  // ForwardGrad but:
  //   - This field is lazily populated.
  //   - This field is a shared_ptr but it must never be
  //     shared by multiple Tensors. See Note [ Using ForwardGrad ]
  // Any transition from not_initialized to initialized
  // must be protected by mutex_
  mutable std::shared_ptr<ForwardGrad> fw_grad_;

  // The hooks_ field is actually reused by both python and cpp logic
  // For both cases, we have a data structure, cpp_hooks_list_ (cpp)
  // or dict (python) which is the canonical copy.
  // Then, for both cases, we always register a single hook to
  // hooks_ which wraps all the hooks in the list/dict.
  // And, again in both cases, if the grad_fn exists on that tensor
  // we will additionally register a single hook to the grad_fn.
  //
  // Note that the cpp and python use cases aren't actually aware of
  // each other, so using both is not defined behavior.
  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  // The post_acc_grad_hooks_ field stores only Python hooks
  // (PyFunctionTensorPostAccGradHooks) that are called after the
  // .grad field has been accumulated into. This is less complicated
  // than the hooks_ field, which encapsulates a lot more.
  std::unique_ptr<PostAccumulateGradHook> post_acc_grad_hooks_ = nullptr;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_{false};

  // Only meaningful on non-leaf variables (must be false otherwise)
  bool retains_grad_{false};

  bool is_view_{false};

  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  uint32_t output_nr_;

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by grad_fn(), grad_accumulator(),
  // fw_grad() and set_fw_grad()
  // This is mutable because we need to be able to acquire this from const
  // version of this class for the functions above
  mutable std::mutex mutex_;

};
```

The most important member of `AutogradMeta` class is `Variable grad_` (in which, `Variable` is alias of `Tensor`). This member is a Tensor which hold gradient result of the parent Tensor.

In summary, a Tensor holds 2 things: (i) Its own data (ii) its gradient