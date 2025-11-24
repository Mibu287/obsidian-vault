#Pytorch #DeepLearning #DeepDive #DataStructure #Dispatcher

# 1. Dispatch Key 

Each `Tensor` has associated dispatch key which can be computed from `dtype`, `layout`, `device`.
In an operation, dispatch key from component tensors are combined into a dispatch key set.
Dispatch key set is used to select kernel for the operation.

## 1.1. Dispatch Key

```C++
enum class DispatchKey : uint16_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // This is not a "real" functionality, but it exists to give us a "nullopt"
  // element we can return for cases when a DispatchKeySet contains no elements.
  // You can think a more semantically accurate definition of DispatchKey is:
  //
  //    using DispatchKey = std::optional<RealDispatchKey>
  //
  // and Undefined == nullopt.  We didn't actually represent
  // it this way because std::optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.

  Undefined = 0,

  // Define an alias for Undefined to represent CatchAll (long term
  // this will get eliminated, but for now it's convenient)
  CatchAll = Undefined,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Functionality Keys ~~~~~~~~~~~~~~~~~~~~~~ //
  // Every value in the enum (up to EndOfFunctionalityKeys)
  // corresponds to an individual "functionality" that can be dispatched to.
  // This is represented in the DispatchKeySet by assigning each of these enum
  // values
  // to each of the remaining (64 - len(BackendComponent)) bits.
  //
  // Most of these functionalities have a single handler assigned to them,
  // making them "runtime keys".
  // That map to a single slot in the runtime operator table.
  //
  // A few functionalities are allowed to be customizable per backend.
  // See [Note: Per-Backend Functionality Dispatch Keys] for details.

  // See [Note: Per-Backend Functionality Dispatch Keys]
  Dense,
  ...

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  EndOfFunctionalityKeys, // End of functionality keys.

// ~~~~~~~~~~~~~~ Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
// Here are backends which you think of as traditionally specifying
// how to implement operations on some device.

  CPU,
  DenseCPU,
  SparseCPU,
  ...

  EndOfRuntimeBackendKeys = EndOfAutogradFunctionalityBackends,
};
```

`DispatchKey` is an enum with functionality and/or backend keys

## 1.2. Dispatch Key Set

```C++
class DispatchKeySet final {
  uint64_t repr_ = 0;
 public:
  constexpr explicit DispatchKeySet(DispatchKey k) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (k == DispatchKey::Undefined) {
      // Case 1: handle Undefined specifically
      repr_ = 0;
    } else if (k <= DispatchKey::EndOfFunctionalityKeys) {
      // Case 2: handle "functionality-only" keys
      // These keys have a functionality bit set, but no backend bits
      // These can technically be either:
      // - valid runtime keys (e.g. DispatchKey::AutogradOther,
      // DispatchKey::FuncTorchBatched, etc)
      // - "building block" keys that aren't actual runtime keys (e.g.
      // DispatchKey::Dense or Sparse)
      uint64_t functionality_val = 1ULL
          << (num_backends + static_cast<uint8_t>(k) - 1);
      repr_ = functionality_val;
    } else if (k <= DispatchKey::EndOfRuntimeBackendKeys) {
      // Case 3: "runtime" keys that have a functionality bit AND a backend bit.
      // First compute which bit to flip for the functionality.
      auto functionality_k = toFunctionalityKey(k);
      // The - 1 is because Undefined is technically a "functionality" that
      // doesn't show up in the bitset. So e.g. Dense is technically the second
      // functionality, but the lowest functionality bit.
      uint64_t functionality_val = 1ULL
          << (num_backends + static_cast<uint8_t>(functionality_k) - 1);

      // then compute which bit to flip for the backend
      // Case 4a: handle the runtime instances of "per-backend functionality"
      // keys For example, given DispatchKey::CPU, we should set:
      // - the Dense functionality bit
      // - the CPUBit backend bit
      // first compute which bit to flip for the backend
      auto backend_k = toBackendComponent(k);
      uint64_t backend_val = backend_k == BackendComponent::InvalidBit
          ? 0
          : 1ULL << (static_cast<uint8_t>(backend_k) - 1);
      repr_ = functionality_val + backend_val;
    } else {
      // At this point, we should have covered every case except for alias keys.
      // Technically it would be possible to add alias dispatch keys to a
      // DispatchKeySet, but the semantics are a little confusing and this
      // currently isn't needed anywhere.
      repr_ = 0;
    }
  }
};
```

`DispatchKeySet` is a bitset for `DispatchKey`
A `DispatchKey` can be added to `DispatchKeySet`
- If `DispatchKey` is functionality-only key ==> The key is added at the corresponding bit of the set
- If `DispatchKey` is functionality-per-backend key ==> Separate functionality / backend keys ==> Add to the corresponding bits of the set

## 1.3. Compute dispatch key

```C++
// This is intended to be a centralized location by which we can determine
// what an appropriate DispatchKey for a tensor is.
inline DispatchKey computeDispatchKey(
  std::optional<ScalarType> dtype,
  std::optional<Layout> layout,
  std::optional<Device> device) {
  ...  
}
```

`DispatchKey` is computed using 3 parameters:
- `dtype`: Data type of the tensor
- `layout`: How data is stored in memory? stride (contiguous) /  sparse / ... ?
- `device`: Which device the data reside? CPU (RAM) / MPS buffer / CUDA device memory / ... ?

# 2. How dispatcher work?

Given a simple addition operation below. How does the operation `+` is called from Python code?

```Python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = x + y
print(z)
```

## 2.1.  Python method signature selector 

```C++
// file: torch/csrc/autograd/generated/python_variable_methods.cpp
// This file is auto-generated from template
// This function is called when Tensor.__add__ method is called by Python
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  // Parse function signatures
  ...
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    ...
  }

  // Select the matched function for the signature
  switch (_r.idx) {
    case 0: {
      ...
      return ...;
    }

    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
```

In essence, Python bindings only do the following things:
- Select the matching function for the function signature
- Release GIL and call the C++ kernel
- Wrap the C++ result in Python Object and return

## 2.2. Finding matching C++ operator 

The Python bindings above call the C++ function below:

```C++
// file: ATen/core/TensorBody.h
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline at::Tensor
Tensor::add( const at::Tensor & other, const at::Scalar & alpha) const {
    return at::_ops::add_Tensor::call(const_cast<Tensor&>(*this), other, alpha);
}
```

The method `Tensor::add` is just a thin wrapper around another function `at::_ops::add_Tensor::call`

```C++
// file: ATen/ops/add_ops.h
// The file is generated from template
struct TORCH_API add_Tensor {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::add")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "Tensor")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};
```

The method call is implemented in file `build/aten/src/ATen/Operators_2.cpp` (which is generated from template)

```C++
// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema> create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}
```

The 2 functions above find registered kernel and execute the kernel. An exception is thrown if no matching kernel found.

At the end of this step, an `OperatorHandle` corresponding the the operation is found

## 2.3. Find matching kernel

```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.h
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
...
public:
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }
...
};
```

```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.h
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return
Dispatcher::call(
  const TypedOperatorHandle<Return(Args...)>& op,
  Args... args
) const {
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);

  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);

    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}
```

Dispatch key set is calculated using the Operator's `dispatchKeyExtractor` instance and all of the arguments' dispatch key set.
The matching kernel is found using the dispatch key set above.
Finally, the kernel is executed using the `call` method.

## 2.4. Execute the kernel

```C++
// file: aten/src/ATen/core/boxing/KernelFunction_impl.h
template<class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
  ...

  auto *functor = boxed_kernel_func_.getFunctor();
  return callUnboxedKernelFunction<Return, Args...>(
         unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...);

  ...
}
```

The kernel is executed and the result is returned.

**NOTE**: When the kernel is executed, Pytorch always check for autograd requirements and re-dispatch after checking. 
Reason: The dispatcher does not know if an operation is part of an execution graph with automatic gradient or not ==> the dispatcher must be conservative and always call autograd kernel first. The selected autograd kernel work with the autograd engine to setup data structure for backward pass later. Then the autograd kernel call `redispatch` to look for the actual kernel for the operation. 
Reminder: Pytorch calls to operations are always dynamic ==> No direct call, always use dispatch/re-dispatch mechanism.
How to make the dispatcher always call autograd kernels? When `Tensor` is initialized `TensorImpl::TensorImpl` is called and `AutogradFunctionality` is added to key set `key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);`

**Example**: For `Add` operation, when the dispatch process finish, the function is selected as below. The function will do the actual work of adding 2 tensors.

```C++
// file: build/aten/src/ATen/UfuncCPUKernel_add.cpp
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);
```

The function take 3 arguments:
- `iter` iterator for all of the tensors
- lambda 1: The function to do element-wise addition
- lambda 2: The function to do vectorized addition
The function will vectorize the input tensors and apply vectorize-function. The remainder will be applied element-wise function.