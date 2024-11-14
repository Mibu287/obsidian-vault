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

