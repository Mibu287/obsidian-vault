#DeepLearning #Pytorch #Miscellaneous #IntrusivePointer #SmartPointer #DataStructure #DeepDive

# 1.  Intrusive pointer

## 1.1. Intrusive pointer target

```C++
// file: c10/util/intrusive_ptr.h
// Note [Stack allocated intrusive_ptr_target safety]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A well known problem with std::enable_shared_from_this is that it
// allows you to create a std::shared_ptr from a stack allocated object,
// which is totally bogus because the object will die once you return
// from the stack.  In intrusive_ptr, we can detect that this has occurred,
// because we set the refcount/weakcount of objects which inherit from
// intrusive_ptr_target to zero, *unless* we can prove that the object
// was dynamically allocated (e.g., via make_intrusive).
//
// Thus, whenever you transmute a T* into a intrusive_ptr<T>, we check
// and make sure that the refcount isn't zero (or, a more subtle
// test for weak_intrusive_ptr<T>, for which the refcount may validly
// be zero, but the weak refcount better not be zero), because that
// tells us if the object was allocated by us.  If it wasn't, no
// intrusive_ptr for you!

// Note [Weak references for intrusive refcounting]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Here's the scheme:
//
//  - refcount == number of strong references to the object
//    weakcount == number of weak references to the object,
//      plus one more if refcount > 0
//    An invariant: refcount > 0  =>  weakcount > 0
//
//  - c10::StorageImpl stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - finalizers are called and data_ptr is deallocated when refcount == 0
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//
//  - When you access c10::StorageImpl via a weak pointer, you must
//    atomically increment the use count, if it is greater than 0.
//    If it is not, you must report that the storage is dead.
//
class C10_API intrusive_ptr_target {
    mutable std::atomic<uint32_t> refcount_;
    mutable std::atomic<uint32_t> weakcount_;
};
```

## 1.2. Intrusive pointer

```C++
template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
    TTarget* target_;
};
```
- Intrusive pointer contain the raw pointer to an instance of intrusive pointer target 
- Utility method `make_intrusive` works like `make_unique` in standard library
    - Check both `refcount_` and `weakcount_` is zero
    - Assign `refcount_` and `weakcount_` to 1
- Utility method `retain_` increase `refcount_` by 1
- Method `reset_` decrement `refcount_` by 1. Check if `refcount_` is zero and `weakcount_` is 1 ==> delete target pointer

## 1.3. Weak intrusive pointer

```C++
template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
    TTarget* target_;
};
```

- Weak intrusive pointer container raw pointer to intrusive pointer target
- Unlike intrusive pointer, weak intrusive pointer increase `weakref_` when utility method `retain_` is called
- Utility method `reset_` decrement `weakcount_` by 1. Check if `weakcount_` is zero ==> delete target pointer

## 1.4. Safety proof

The protocol for intrusive pointer is safe:
- At T = 0, both `refcount_` and `weakcount_` equals 1
- At T = k, assume there are m strong and n weak pointers (m >= 0, n >= 0, m + n > 0) ==> refcount_ == m and weakcount_ == n + 1

Invariants:
- At any point in time, if a pointer P is deconstructed and P is not the last pointer ==> P's destructor does not delete target. Proof:
    - If P is strong pointer
        - If there is strong pointer alive when refcount_ is checked ==> `refcount_` > 0
        - If there is weak pointer alive when `weakcount_` is checked ==> `weakcount_` > 1
     - If P is weak pointer
        - If there is strong pointer alive when `weakcount_` is checked ==> `weakcount_` > 0 
        - If there is weak pointer alive when `weakcount_` is checked ==> `weakcount_` > 0

- If pointer P is the pointer ==> P's destructor will delete the target. Proof:
    - If P is strong pointer
        - There is no strong pointer alive when `refcount_` is checked ==> `refcount_` = 0
        - There is no weak pointer alive when `weakcount_` is checked ==> `weakcount_` = 0
        - ==> Delete target
    - If P is weak pointer
        - There is no strong and weak pointer alive when `weakcount_` is checked ==> Delete target

- In summary:
    - Only the last strong pointer or weak pointer can delete target
    - If the last strong pointer is the last to hold `weakcount_` ==> it delete the target
    - If the last weak pointer is the last to hold `weakcount_` ==> it delete the target
        == > There is exactly 1 pointer whose destructor delete the target