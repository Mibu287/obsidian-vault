#DeepLearning #Pytorch #DataStructure #DeepDive #DeepLearning #Miscellaneous 

# 1. Definition

```C++
class C10_API TypeMeta final {
  uint16_t index_;
};
```

`TypeMeta` class is a wrapper around an unsigned integer of size 16 bits.

Basically, `TypeMeta` is an index of actual `TypeMetaData` in an array. `TypeMeta` can be viewed as a pointer to `TypeMetaData`. The array `TypeMetaData` is setup at program start-up by Pytorch's registration API.

The class `TypeMetaData` contains basic information about a type like
- The size
- The name
- How to construct
- How to destruct
- Type ID for internal use. Pytorch compute type id using CRC64 function on the type's full name

The definition of `TypeMetaData` as belows:

```C++
struct TypeMetaData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  size_t itemsize_;
  New* new_;
  PlacementNew* placementNew_;
  Copy* copy_;
  PlacementDelete* placementDelete_;
  Delete* delete_;
  TypeIdentifier id_;
  c10::string_view name_;
};
```
# 2.  Accessing underlying data

```C++
inline const detail::TypeMetaData& data() const {
  return typeMetaDatas()[index_];
}
```

As note above, `TypeMeta` is practically a pointer to `TypeMetaData` ==> underlying `TypeMetaData` is retrieved by using the index member to access the global array of `TypeMetaData`. 
