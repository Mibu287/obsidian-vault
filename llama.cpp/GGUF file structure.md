
#DeepLearning  #llamacpp #DataStructure

# 1. Overview

![model.gguf|653](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png)

**Design**:
- 24 first bytes of a GGUF are always there. It dictate what come next in the file
- The file always contains information about the size of the data to be read next. E.g: 
    - Number of metadata key-value pairs is in byte offset 16:23 ==> reader will attempt to read exactly that amount of key-value pairs.
    - When reader attempt to read a string, it must read next 8 bytes to determine length of the string. Then, the reader will read exactly that amount and interpret it as a string.
    - Before reading tensor data, the reader must read file header for offset, shape, type of the tensor. With the information, the reader can deduce where the tensor is stored in file, how many bytes to read or mmap.
-  

**Structure**:

- 1st 4 bytes: GGUF magic number
- 2nd 4 bytes: GGUF version
- 