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