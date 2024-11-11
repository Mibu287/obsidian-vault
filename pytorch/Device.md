# 1. Device Type

```C++
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  ...
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
}
```

# 2. Device

```C++
// file: c10/core/Device.h
/// Represents a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct C10_API Device final {
  DeviceType type_;
  DeviceIndex index_ = -1;
};
```

`Device` is just a thin structure which contain device type and index of the device (Pytorch support multiple devices ==> Needs device index)

Currently (11-Nov-2024), Pytorch supports about 20 device types.