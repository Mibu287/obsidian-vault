#DeepLearning #Pytorch #DeepDive 
## 1. What is a Tensor?
- a Tensor is a multi-dimensional array of single data type
  e.g:
```text
  Tensor([
	[1, 2],
	[2, 3]
  ]), dtype=float32
```
- What can a Tensor do?
	- Arithmetic operations with other tensors / scalers
## 2. Tensor in Pytorch
- Python object, C++ implementation (ATen library)

- Interaction with other Python array library (numpy)

```python
np_array = np.ones((2, 2))
torch_array = torch.tensor(np_array) # New copy is made
torch_array = torch.from_numpy(np_array) # A ref to np array is made
```

- Implementation of ```from_numpy``` method:
```cpp
at::Tensor tensor_from_numpy(PyObject* obj, ...) {
	auto array = (PyArrayObject*)obj;
	int ndim = ...,
	auto sizes = ...;
	auto strides = ...;
	void* data_ptr = PyArray_DATA(array);
	Py_INCREF(obj);
	return at::lift_fresh(at::from_blob(
		data_ptr, sizes, strides, ...
	))
}
```
- Tensor storage
	- Abstraction to separate raw data and its interpretations.
		-e.g: 2 tensors share the same storage but has different views
	- Memory allocators:
		-CPU and/or GPU
		-PyTorch use the abstract class Allocator
	- Each Tensor has ```Storage *storage``` field which is pointer to Storage object
		- The Storage object is a pointer to the raw data and allocator object
- Tensor trinity
	- Device: Which device the tensor reside
	- Layout: How data is stored in the tensor (strided, sparse, mkldnn, ...)
	- dtype: Type of the data inside the tensor (1 dtype / tensor)
- Each tensor is a 3-tuple of the 3 properties (Device, Layout, dtype). Althought, not all tensor has repective kernels.

## 3. Exploring Pytorch's source code about Tensor
### 3.1. Tensor definition
#### 3.1.1. class Tensor
```C++
// file: aten/src/ATen/core/TensorBody.h
// The source file is auto-generated from template
class TORCH_API Tensor: public TensorBase {
	// No additional data-member compared to TensorBase
	// Only method definitions
	...
}
```

#### 3.1.2. class TensorBase
```C++
// file: aten/src/ATen/core/TensorBase.h
// Tensor, being the central data structure in PyTorch, gets used and
// it's header included almost everywhere. Unfortunately this means
// every time an operator signature is updated or changed in
// native_functions.yaml, you (and every other PyTorch developer) need
// to recompile all of ATen and it's dependencies.
//
// TensorBase aims to break up these header dependencies, and improve
// incremental build times for all PyTorch developers. TensorBase
// represents a reference counted handle to TensorImpl, exactly the
// same as Tensor. However, TensorBase doesn't have code generated
// methods in it's API and thus no dependence on native_functions.yaml.
class TensorBase {
protected:
	// TensorBase is just ref-counted pointer to TensorImpl
	c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
}
```

#### 3.1.3. struct TensorImpl
```C++

```