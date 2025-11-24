#Pytorch #DeepLearning #DeepDive #KernelRegistration

# 1. The Dispatcher classes

## 1.1 Dispatcher

```C++
/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 * Most end users shouldn't use this directly; if you're trying to register
 * ops look in op_registration
 */
class TORCH_API Dispatcher final {
  struct OperatorDef final {
    explicit OperatorDef(OperatorName&& op_name)
    : op(std::move(op_name)) {}

    impl::OperatorEntry op;

    // These refer to the number of outstanding RegistrationHandleRAII
    // for this operator.  def_count reflects only def() registrations
    // (in the new world, this should only ever be 1, but old style
    // registrations may register the schema multiple times, which
    // will increase this count).  def_and_impl_count reflects the number
    // of combined def() and impl() registrations.  When the last def() gets
    // unregistered, we must immediately call the Deregistered listeners, but we
    // must not actually delete the handle as there are other outstanding RAII
    // destructors which will try to destruct and they had better still have a
    // working operator handle in this case
    size_t def_count = 0;
    size_t def_and_impl_count = 0;
  };

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  // Map from namespace to debug string (saying, e.g., where the library was defined)
  ska::flat_hash_map<std::string, std::string> libraries_;

  std::array<impl::AnnotatedKernel, num_runtime_entries> backendFallbackKernels_;

  std::unique_ptr<detail::RegistrationListenerList> listeners_;

  // This condition variable gets notified whenever we add a new def/impl to the
  // dispatch table.  This is primarily used by multipy/torchdeploy, when
  // we have multiple interpreters trying to register to the dispatch table.
  // In this situation, whenever the non-primary interpreter would have tried
  // to register to the dispatch table, instead it will check to see if the
  // expected registration has already been made, and if it hasn't, wait on
  // this condition variable to see if it was just racing with the primary
  // interpreter.
  //
  // We expect it to be rare for there to be any waiters on this condition
  // variable.  This is mostly just to help give better diagnostics if
  // something goes horribly wrong
  std::condition_variable cond_var_;

  // Protect concurrent access to the dispatcher.  We store this in a
  // `shared_ptr` as we return callbacks that call back into dispatcher methods,
  // and we need to be able to handle and guard against the event when the
  // `Dispatcher` has been destroyed before the callbacks fire.
  std::shared_ptr<Guard> guard_;
};
```

The `Dispatcher` contains all of the kernels which were registered using the registration API.
The 2 important objects in Dispatcher are:
  - `operators_` which is a list of all registered operators
  - `operatorLookupTable_` which is a lookup table mapped operator name and its implementation

## 1.2. Operator Entry

```C++
// Internal data structure that records information about a specific operator.
// It's not part of the public API; typically, users will interact with
// OperatorHandle instead.
//
// Concurrent writes to OperatorEntry are protected by the GLOBAL Dispatcher
// lock (this is important because some methods in OperatorEntry access
// dispatcher state)
class TORCH_API OperatorEntry final {
  OperatorName name_;
  std::optional<AnnotatedSchema> schema_;
  #ifndef C10_MOBILE
    std::vector<at::Tag> tags_;
  #endif
  std::array<KernelFunction, c10::num_runtime_entries> dispatchTable_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  // Pointer to the torch.ops.ns.op.overload object for speed
  c10::PyHandleCache py_cache_;

  // kernels_ stores all registered kernels for the corresponding dispatch key
  // and catchAllKernels_ stores the catch-all kernels.
  // If an operator library gets loaded that overwrites an already existing kernel,
  // both kernels will be in that list but only the newer one will be in
  // dispatchTable. If any of the kernels go away (say the library gets
  // unloaded), we remove the kernel from this list and update the
  // dispatchTable if necessary.
  // Kernels in the list are ordered by registration time descendingly,
  // newer registrations are before older registrations.
  // We do not combine dispatchTable and kernels into one hash map because
  // kernels is a larger data structure and accessed quite infrequently
  // while dispatchTable is accessed often and should be kept small to fit
  // into CPU caches.
  // Invariants:
  //  - dispatchTable[dispatch_key] == kernels_[dispatch_key].front()
  //  - dispatchTable[dispatch_key] does not exist if and only if
  //    kernels_[dispatch_key] does not exist
  //  - If kernels_[dispatch_key] exists, then it has elements.
  //    It is never an empty list.
  //
  // Why do we do that?
  // -----
  // We mostly do this to enable Jupyter notebooks where a cell registering
  // a kernel could be executed multiple times and the later execution
  // should overwrite the earlier one. Note that this still fails when the
  // function schema changed between the executions, but it works as long
  // as the function schema didn't change. A better solution would be to
  // unload the old extension library from the Jupyter cell when the cell is
  // re-executed and then only allow one kernel here, i.e. error if a kernel
  // is already registered, but that's a lot of effort to implement and
  // currently not high-pri.
  ska::flat_hash_map<DispatchKey,
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
                     // On mobile, we needn't worry about Jupyter notebooks.
                     std::array<AnnotatedKernel, 1>
#else
                     std::list<AnnotatedKernel>
#endif
                     > kernels_;

  // cpp_signature_ stores function signature if any of
  // the kernels was created in a way that allowed us to know the function
  // signature (i.e. by supplying an unboxed C++ kernel function).
  // If this is set, it will be used to check that future kernel
  // registrations match and it will be used in unboxed function calls
  // to verify their arguments against the known function signature.
  struct CppSignatureWithDebug {
    CppSignature signature;
    std::string debug;
    std::optional<DispatchKey> dispatch_key;
  };
  std::optional<CppSignatureWithDebug> cpp_signature_;
  std::optional<CppSignatureWithDebug> sym_cpp_signature_;

  // A Python custom error handler for OperatorEntry::reportError
  std::unique_ptr<c10::SafePyObject> report_error_callback_;

  // Whether this operator needs to be observed with RecordFunction
  const bool is_observed_;
};
```

The `OperatorDef` contains some structures as below:
- `name_` - the name of the operator
- `schema_` - the annotated schema of the operator (optional)
- `dispatchTable_` - the array which hold pointers to kernel functions for every backends and functionalities
- `dispatchKeyExtractor` -  knows how to get a dispatch key given a list of arguments for an operator call

Operator Handle

```C++
/**
 * This is a handle to an operator schema registered with the dispatcher.
 * This handle can be used to register kernels with the dispatcher or
 * to lookup a kernel for a certain set of arguments.
 */
class TORCH_API OperatorHandle {
  Dispatcher::OperatorDef* operatorDef_;

  // We need to store this iterator in order to make
  // Dispatcher::cleanup() fast -- it runs a lot on program
  // termination (and presuambly library unloading).
  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};
```

Basically, `OperatorHandle` is a pointer to `OperatorDef`. `OperatorDef` is a thin wrapper of `OpertorEntry` which contains actual dispatch information for the operation.

## 1.3. Dispatch Key Extractor

```C++
/**
 * An instance of DispatchKeyExtractor knows how to get a dispatch key given
 * a list of arguments for an operator call.
 *
 * The instance is specific for a certain operator as:
 *  - In boxed dispatch, different operators have different ways to extract
 *    the dispatch key (e.g. different numbers of arguments), and we precompute
 *    the stack locations we should look at; and
 *  - In all dispatch, some backends should be excluded from dispatch because
 *    they have been registered as fallthrough.  The set of excluded backends
 *    varies from operator, as some operators may have overridden the
 *    fallthrough with custom behavior.
 *
 *   Note - this should maintain identical impl to the py dispatcher key extraction logic
 *   at pytorch/torch/dispatcher.py
 */
struct TORCH_API DispatchKeyExtractor final {
  // this is a bitset that has ones for each argument index which has to be
  // considered for dispatch. This avoids having to iterate over the stack
  // to find all the tensors. The bits are stored in reverse order, i.e.
  // dispatch_arg_indices_reverse_[i] == true, then the i-th argument from
  // the top of the stack (i.e. the i-th last argument of the function)
  // is relevant for dispatch.
  // dispatch_arg_indices_reverse_ is allowed to have zero bits set; that just means you must do the
  // fallthrough
  c10::utils::bitset dispatch_arg_indices_reverse_;

  // Set of functionality keys for which the operator does NOT have fallthrough kernel.
  DispatchKeySet nonFallthroughKeys_;
  // Set of functionality keys for which the operator does NOT have fallthrough kernel, defined PER BACKEND.
  // This is only needed if we know that the operator has a different set of fallthroughs defined for some backends.
  std::array<DispatchKeySet, num_backends> nonFallthroughKeysPerBackend_;
  // Flag to tell us if we can use the single set of nonFallthroughKeys_ (fast path),
  // or if we need to fall back to the slower path and check nonFallthroughKeysPerBackend_
  bool requiresBitsetPerBackend_;
};
```

# 2. Registration

## 2.1. Register definition of operators

An operator's definition is registered by using the macro `TORCH_LIBRARY`

```C++
/// Macro for defining a function that will be run at static
/// initialization time to define a library of operators in the
/// namespace `ns` (must be a valid C++ identifier, no quotes).
/// Use this macro when you want to define a new set of custom operators
/// that do not already exist in PyTorch.
///
/// Example usage:
///
/// ```
/// TORCH_LIBRARY(myops, m) {
///   // m is a torch::Library; methods on it will define
///   // operators in the myops namespace
///   m.def("add", add_impl);
/// }
/// ```
///
/// The `m` argument is bound to a torch::Library that is used to
/// register operators.  There may only be one TORCH_LIBRARY()
/// for any given namespace.
#define TORCH_LIBRARY(ns, m)\
  static void TORCH_LIBRARY_init_##ns(torch::Library&);\
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF, \
      &TORCH_LIBRARY_init_##ns, \
      #ns, \
      std::nullopt, \
      __FILE__, \
      __LINE__); \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)
```

Explanation:
- The `TORCH_LIBRARY` macro define a static variable `TORCH_LIBRARY_static_init_##ns` e.g `TORCH_LIBRARY_static_init_aten`.
- The variables are declared as `static` ==> No other translation unit can read/write to the variables. In fact, the variables are only defined so registration work can be done at static initialization time.
- The definition of `TorchLibraryInit` as following:

```C++
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      std::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};
```

- At static initialization time, `TorchLibraryInit` is initialized. Its constructor create a `Library` object and then call the function `TORCH_LIBRARY_init_##ns` with the `Library` as an argument. Example: definition of `TORCH_LIBRARY_init_aten`:

```C++
// file: build/aten/src/ATen/RegisterSchema.cpp
namespace at {
TORCH_LIBRARY(aten, m) {
  ...
  m.def("add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", tags_5);
  ...
};
}
```

- In the snippet above, declaration for add operation is registered. The meat of the `Library::def` method as below:

```C++
// file: aten/src/ATen/core/library.cpp
Library& Library::_def(c10::FunctionSchema&& schema, ...) {
  ...
  c10::Dispatcher::singleton().registerDef(
    std::move(schema),
    debugString(file_, line_),
    tags
  )
  ...
}
```

- Definition of `Dispatcher::registerDef` method:
```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  ...

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));

  ...
}
```

```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.cpp
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != std::nullopt) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  ...
  operatorLookupTable.emplace(op_name, handle);
  ...
}
```

- In the snippet above, the `Dispatcher` find or create new operator entry object. Then method `registerSchema` is called on the operator entry object.

```C++
// file: aten/src/ATen/core/dispatch/OperatorEntry.cpp
void OperatorEntry::registerSchema(FunctionSchema&& schema, ...) {
  ...
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
  ...
}
```

**Summary**
- Macro `TORCH_LIBRARY` is used to register function schema
- `TORCH_LIBRARY` create a `Library` instance at static initialization time. `Library` object provides API for defining operators and implementations.
- When `TORCH_LIBRARY` is used,  a trailing function body is defined. In which, method `def` is called on the `Library` object. E.g. `m.def("add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");`
- The operator signature is parsed to create a `FunctionSchema` object
- `Dispatcher::registerDef` is called on the `Dispatcher` singleton object. The method find or create new `OperatorEntry`
- The newly created `OperatorEntry` contains only the name and schema. Its implementations is currently blank

## 2.2. Register operator's  implementations

```C++
// file: torch/library.h

#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

#define _TORCH_LIBRARY_IMPL(ns, k, m, uid) \
  static void C10_CONCATENATE( \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);\
  static const torch::detail::TorchLibraryInit C10_CONCATENATE( \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)( \
      torch::Library::IMPL, \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k) \
           ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid \
           : [](torch::&) -> void {}),\
      #ns, \
      std::make_optional(c10::DispatchKey::k), \
      __FILE__, \
      __LINE__);\
  void C10_CONCATENATE(\
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)
```

Like `TORCH_LIBRARY`, the macro `TORCH_LIBRARY_IMPL` define a static variable to:
- Do some work (register kernels, ...) at static initialization time
- Do clean up when the library is unloaded

The main parts of the registration process is like the following:

```C++
// 1. If the operator's dispatch key is not in allow list
// Return a do-nothing function.
// Otherwise, return the function which user about to define.
auto f = (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)
          ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid
          : [](torch::&) -> void {});

// 2. Define a new library
// A library provide APIs for registering defintions and implementations
Library m {
    torch::Library::IMPL,
    new_func,
    ns,
    c10::DispatchKey::k,
    __FILE__,
    __LINE__
};

// 3. Register an implementation
m.impl("add.Tensor", TORCH_FN(wrapper_CPU_add_Tensor));
```

Details of how kernel registration work as following:

```C++
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);

  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;

  c10::Dispatcher::singleton().registerImpl(
    std::move(name),
    dispatch_key,
    std::move(f.func_),
    f.cpp_signature_,
    std::move(f.schema_),
    debugString(std::move(f.debug_), file_, line_)
  );
  return *this;
}
```

```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  ...
  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;
  ...
}
```

```C++
// file: aten/src/ATen/core/dispatch/OperatorEntry.cpp
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  ...

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  // Redirect catchAll registrations to CompositeImplicitAutograd.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];


  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));

  AnnotatedKernelContainerIterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}
```

**Summary**
- The macro `TORCH_LIBRARY_IMPL` is called and a function body is defined with the invocation of this macro. Inside the function body, method `Library.impl` is called to register a kernel. E.g `m.impl("add.Tensor",TORCH_FN(wrapper_CPU_add_Tensor));` register add kernel to the dispatcher.
- The kernel signature supplied to `Library::impl` is parsed. An `OperationDef` is retrieved or created using the operation name in the signature.
- `DispatchKey` is calculated using the kernel's dispatch key and the `OperationDef`'s dispatch key. The kernel's key is preferred if present.
- The kernel is added to (i) the `kernels_` hash table and (ii) `dispatchTable_` array

Refer to [[Kernels]] for more information about Pytorch's data structures for kernels.

# 3. Searching

```C++
// file: aten/src/ATen/core/dispatch/Dispatcher.cpp
// The function below looking for registered schema in Dispatcher singleton object
OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    // Check if we have ANYTHING; if that's the case, that means you're
    // missing schema
    auto it2 = findOp({name, overload_name});
    if (!it2.has_value()) {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name);
    } else {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name,
        " but we found an implementation; did you forget to def() the operator?");
    }
  }
  return it.value();
}

std::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return std::nullopt;
    }
    return found->second;
  });
}
```

Operator definitions are stored in a list, operator handles are stored in a hash map.
While finding kernel, the dispatcher use schema name to look up in operator lookup table.
