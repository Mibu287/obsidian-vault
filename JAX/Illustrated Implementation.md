#JAXML 

Reference link [Autodidax](https://docs.jax.dev/en/latest/autodidax.html) 

# 1. Main concepts

  - Trace: The classes contain logic to do specific jobs.
    Examples:
     - Evaluation trace: Do actual computation on each operation
     - Grad trace: Do computation and reverse gradients on inputs
     - JIT trace: JIT compile the user-defined functions before do actual computation
    
 - Tracer: The object being traced.
   Examples:
     - A Python float
     - A JAX ndarray
     - Other tracer (tracers are composable). E.g: JIT tracer hold grad tracer.

# 2. How it works?

```python
from jax import numpy as np

def f(x: np.ndarray) -> np.ndarray:
    y = jax.cos(x)
    z = jax.sin(y)
    return z

g = jax.grad(f)
h = jax.vmap(g)
j = jax.jit(h)

x = ...
j(x)
```

Logically, when `j` is called with `x` as argument, it JIT-compiled the function `h` and called it. the JIT-compiled function `h` send each element at dimension 0 of `x` to function `g`. `g` keep track of the computation of function `f` and the corresponding reversed gradients.

But, how JAX keep track of the computation to do JIT-compilation and reversed gradient calculations?

- When `j` is called with argument `x: np.ndarray`, it wrap `x` in a tracer object. Let's called the tracer object `JitTracer`
- `j` then call `h` with the `JitTracer` object as argument. `h` wrap the object in the new tracer object. Let's call it `VMapTracer`
- `h` then call `g` with the `VMapTracer` as argument. `g` wrap the object in the new tracer object. Let's call it `GradTracer`
- `g` then call `f` with `GradTracer` as argument.  At each step of the function `f`, the tracer override the corresponding operation and perform its own logic (E.g. build computation plan, ...). Then, the tracer get its inner tracer and do the same operation on the inner tracer. ==> Each operation is recursively done with each tracer object object ==> Tracers are composable ==> Function transformations are composable.

# 3. Notes

- JAX use trace mechanism for its function transformation logic ==> the Python interpreter do heavy lifting. JAX throw its tracer objects to Python interpreter, receive the output tracer and do analysis on the output tracer. JAX does not need to understand the inner working of user-defined functions in order to do transformation.
- Because of the trace mechanism, JAX require user-defined function to be 'pure' in order to function accurately. Consider the example:

```python
from jax import numpy as np

counter = 0

def f(x: np.ndarray) -> np.ndarray:
    global counter
    counter += 1
    return jax.sin(x * counter)

j = jax.jit(f)

x = ...
j(x)
```

`j` do tracing on `f`. At the time of tracing, `counter` variable = 0 ==> the output tracer object record counter = 0. The output tracer contains no information about counter is global variable which can be changed at each iteration. Only the Python interpreter has this information.

==> `j` take a snapshot of `f` at this iteration and do JIT-compilation.

==> if `f` is pure (i.e: the result only depends on its inputs. No side effects occur). `j` is guaranteed to always output the same result as `f`. If not, sometimes the outputs may be different.

- Control flow

```python
from jax import numpy as np

def f(x: np.ndarray, cond: bool) -> np.ndarray:
    if cond:
        ...
    else:
        ...

x = ...

f(x)          # Ok
jax.jit(f)(x) # Failed
```

if `jax.jit` allow `f` to be compiled, it can only capture `f` at 1 in 2 branches
==> when the JITed function is called with other `cond`, it may give the wrong result.
==> JAX throw error when any tracer is used in flow-control construct to prevent this logic error.

To fix this error:

1. Declare `cond` as static argument, JAX will produce the compiled version in accordance to only 1 value of `cond`. If `cond` changed, the function will be compiled again.
2. Use JAX control flow primitives, i.e. function-like control flow. E.g. `lax.cond`, `lax.while_loop`, `lax.fori_loop`