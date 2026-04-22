
#DeepLearning  #llamacpp #DataStructure

# 1. Overview

![model.gguf|653](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png)

**Design**:
- 24 first bytes of a GGUF are always there. It dictate what come next in the file
- The file always contains information about the size of the data to be read next. E.g: 
    - Number of metadata key-value pairs is in byte offset 16:23 ==> reader will attempt to read exactly that amount of key-value pairs.
    - When reader attempt to read a string, it must read next 8 bytes to determine length of the string. Then, the reader will read exactly that amount and interpret it as a string.
    - Before reading tensor data, the reader must read file header for offset, shape, type of the tensor. With the information, the reader can deduce where the tensor is stored in file, how many bytes to read or mmap.
- GGUF file currently require the system write the file and the system read the file has the same endianness. Any difference in endianness may not be caught and may lead to unexpected errors. Due to this assumption, GGUF reader simply read number from the file to memory (like memcpy).

**Structure**:

- 1st 4 bytes: GGUF magic number
- 2nd 4 bytes: GGUF version
- Next 8 bytes: tensor count
- Next 8 bytes: number of key-value pairs
- Next section: key-value pairs.
    - key: read 8 bytes to get key length, then read the content.
    - value: read 4 bytes to determine data type. Then depends on data type, read the content.
    - Move on to the next key-value pair
- Next section: tensor info. The section contain all metadata of all the tensors (i.e. name, shape, type, offsets, ...). The information is packed together. The GGUF reader must read information about each tensor sequentially until the end.
- Data section: This section contain actual data of all tensors. Based on metadata of the previous section, reader seek to the tensor to read its data. Note: tensor data are numbers (int, float, double, mixed, ...). Tensor data are store continuously like how it is stored in memory ==> reader can mmap the file region to memory and work on it (no conversion needed). 