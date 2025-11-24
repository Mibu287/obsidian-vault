#Pytorch #DeepDive #DataStructure 

# 1. Kernel Function

```C++
/**
 * KernelFunction is similar to std::function but stores a kernel function.
 * You can create a KernelFunction from a boxed or unboxed function/functor/lambda
 * and call it in a boxed or unboxed way. If the way it was created doesn't
 * match the way it was called, it will do boxing or unboxing as necessary.
 */
class TORCH_API KernelFunction final {
 public:
  using InternalBoxedKernelFunction = BoxedKernel::InternalBoxedKernelFunction;
  using BoxedKernelFunction = BoxedKernel::BoxedKernelFunction;
  using BoxedKernelFunction_withDispatchKeys = BoxedKernel::BoxedKernelFunction_withDispatchKeys;

  /**
   * Call the function in a boxed way.
   * If the kernel function was created with an unboxed function,
   * this will call an unboxing wrapper which then calls into that
   * unboxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.callBoxed(stack);
   *
   * Or, with an unboxed implementation:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callBoxed(stack);
   */
  void callBoxed(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Stack* stack) const;

  /**
   * Call the function in an unboxed way.
   * If the kernel function was created with a boxed function,
   * this will box all inputs and then call into that boxed function.
   *
   * Note that this doesn't work for all types yet.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   *
   * Or, with a boxed implementation:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   */
  template<class Return, class... Args>
  Return call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const;

 private:
  BoxedKernel boxed_kernel_func_;
  void* unboxed_kernel_func_;
  void* sym_unboxed_kernel_func_;
};
````

A `KernelFunction` is an abstraction over all type of kernel functions (boxed or unboxed). The inner kernel can be invoked using `callBoxed` or `call` methods.

# 2. Boxed Kernel

```C++
/**
 * BoxedKernel is similar to a std::function storing a boxed kernel.
 */

using InternalBoxedKernelFunction = void(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);
using BoxedKernelFunction = void(const OperatorHandle&, Stack*);
using BoxedKernelFunction_withDispatchKeys = void(const OperatorHandle&, DispatchKeySet, Stack*);
  
class TORCH_API BoxedKernel final {
  c10::intrusive_ptr<OperatorKernel> functor_;
  InternalBoxedKernelFunction* boxed_kernel_func_;
};
```

# 3. CPP Function

```C++
// file: torch/library.h
// This class erases the type of the passed in function, but durably records the type via an inferred schema for the function.
class TORCH_API CppFunction final {
  std::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  std::optional<c10::impl::CppSignature> cpp_signature_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;
};
```

`KernelFunction` class erase type from C++ function. `CppFunction` store function schema in its structure.

```C++
// file: aten/src/ATen/core/function_schema.h
struct TORCH_API FunctionSchema {
  OperatorName name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primitive' operators whose
  // arguments are not checked by schema
  bool is_vararg_;
  bool is_varret_;

  // if no alias information is directly specified, what kind of "default"
  // alias information should we infer?
  // NB: due to alias analysis kind merging, this may be nullopt.  Eventually
  // this should always be set no matter what
  std::optional<AliasAnalysisKind> alias_kind_;
};


struct TORCH_API Argument {
  std::string name_;
  TypePtr type_;
  TypePtr real_type_; // this is ScalarType, not int, e.g.
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  std::optional<int32_t> N_;

  std::optional<IValue> default_value_;
  // AliasInfo is huge, so let's only allocate memory for it if
  // necessary (which it isn't during schema parsing on startup, to
  // give a pertinent example).
  std::unique_ptr<AliasInfo> alias_info_;
  // is this only specifiable as a keyword argument?
  bool kwarg_only_;
  // marks if the argument is out variant of the schema
  bool is_out_;
};
```

```C++
// file: aten/src/ATen/core/dispatch/CppSignature.h
class TORCH_API CppSignature final {
  std::type_index signature_;
};
```

`FunctionSchema` store more information about actual signatures of the function. `CppFunction` only store the type-index of the C++ function ==> `CppFunction` can compare different C++ function type and retrieve the name of the functions.

==**NOTE:**==
- For unboxed operators, the C++ functions are stored as type-erased objects. However, each operator can cast the type-erased objects back into the typed functions.
- For boxed operators, all arguments are put on a stack, the operators take arguments from the stack and push the returned value back on the stack.