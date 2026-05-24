
#llamacpp #DeepLearning #LLM

### GPT2 Configuration

- `n_embd_head` = 12
    
- `n_layers` = 12
    

### `build_inp_embd (ggml_tensor* tok_embd)`

- Create a tensor to hold the input token
    
    - For GPT2 models, it is of size `[1, 1, 1, 1]` of type `i32` (token id.)
        
- Create a tensor to hold the embedding vector for the input token
    
    - For GPT2, it is of size `[768, 1, 1, 1]` of type `F32`.
        
- Create a new tensor which is the result of a `get rows` operation between:
    
    - Input token tensor
        
    - Token embedding matrix (Loaded from the model file)
        

~~`build_attn_inp_kv`~~

### Build Positional Embedding Tensor

$\rightarrow$ It is the result of the `get row` function using:

- Positional embedding matrix (loaded from file)
    
- Input position vector.
    
- Add the token embedding + positional embedding together.
    

### Attention Layers

**Loop over `n_layer` to build attention layers.**

**Overall flow:**

1. Normalize input.
    
2. Calculate Attention:
    $$\text{Attn} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$
3. Add Attn to input.
    
4. Feed forward layer:
    
    - Normalize input.
        
    - Feed forward.
        
5. Add to input vector.
    

$\rightarrow$ Feed to the next layer