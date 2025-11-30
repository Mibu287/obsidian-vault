
JAX library heavily utilize functional programming techniques. E.g. higher-order functions, currying, function transformation, ...

# 1.  function `curry`

```python
def curry(f):
  """Curries arguments of f, returning a function on any remaining arguments.
  """
  return wraps(f)(partial(partial, f))

def wraps(
    wrapped: Callable,
    namestr: str | None = None,
    docstr: str | None = None,
    **kwargs,
) -> Callable[[T], T]:
  """
  Like functools.wraps, but with finer-grained control over the name and docstring
  of the resulting function.
  """
  def wrapper(fun: T) -> T:
    try:
      name = fun_name(wrapped)
      doc = getattr(wrapped, "__doc__", "") or ""
      fun.__dict__.update(getattr(wrapped, "__dict__", {}))
      fun.__annotations__ = getattr(wrapped, "__annotations__", {})
      fun.__name__ = name if namestr is None else namestr.format(fun=name)
      fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
      fun.__doc__ = (doc if docstr is None
                     else docstr.format(fun=name, doc=doc, **kwargs))
      fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
      fun.__wrapped__ = wrapped
    except Exception:
      pass
    return fun
  return wrapper
```

`curry` transform a function into 2-level partial function. i.e. the returned function can be called in 2 steps:

1. Step 1: Called with a number of arguments ==> Return a partial function object
2. Step 2: The partial function object from step 1 is called with the remaining arguments of f ==> It called the function with concatenation of all arguments


# 2. class `WrappedFun`

```python
class WrappedFun:
  """Represents a function `f` to which `transforms` are to be applied.

  Args:
    f: the function to be transformed.
    f_transformed: transformed function.
    transforms: a tuple of `(gen, gen_static_args)` tuples representing
      transformations to apply to `f.` Here `gen` is a generator function and
      `gen_static_args` is a tuple of static arguments for the generator. See
      description at the start of this module for the expected behavior of the
      generator.
    stores: a list of out_store for the auxiliary output of the `transforms`.
    params: a tuple of `(name, param)` tuples representing extra parameters to
      pass as keyword arguments to `f`, along with the transformed keyword
      arguments.
    in_type: optional input type
    debug_info: debugging info about the function being wrapped.
  """
  __slots__ = ("f", "f_transformed", "transforms", "stores", "params", "in_type", "debug_info")

  f: Callable
  f_transformed: Callable
  transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...]
  stores: tuple[Store | EqualStore | None, ...]
  params: tuple[tuple[str, Any], ...]
  in_type: core.InputType | None
  debug_info: DebugInfo

  def wrap(self, gen, gen_static_args,
           out_store: Store | EqualStore | None) -> WrappedFun:
    """Add another transform and its store."""
    if out_store is None:
      return WrappedFun(self.f, partial(gen, self.f_transformed, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)
    else:
      return WrappedFun(self.f, partial(gen, self.f_transformed, out_store, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)

  def populate_stores(self, stores):
    """Copy the values from the `stores` into `self.stores`."""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """Calls the transformed function"""
    return self.f_transformed(*args, **kwargs)
```

a `WrappedFun` contain a function and its transformed function. It has `wrap` method to wrap one more transformation.

# 3. function `wraps`

```python
def wraps(
    wrapped: Callable,
    namestr: str | None = None,
    docstr: str | None = None,
    **kwargs,
) -> Callable[[T], T]:
  """
  Like functools.wraps, but with finer-grained control over the name and docstring
  of the resulting function.
  """
  def wrapper(fun: T) -> T:
    try:
      name = fun_name(wrapped)
      doc = getattr(wrapped, "__doc__", "") or ""
      fun.__dict__.update(getattr(wrapped, "__dict__", {}))
      fun.__annotations__ = getattr(wrapped, "__annotations__", {})
      fun.__name__ = name if namestr is None else namestr.format(fun=name)
      fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
      fun.__doc__ = (doc if docstr is None
                     else docstr.format(fun=name, doc=doc, **kwargs))
      fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
      fun.__wrapped__ = wrapped
    except Exception:
      pass
    return fun
  return wrapper
```

Function `wraps` change meta data of the wrapper to match that of the wrapped.
`wraps` does not change any behaviors of the wrapper function.

Why? Hide implementation details so readers do not need to care about the wrapper.

# 4. function `api_boundary`

Call the wrapped function inside a try-catch statement for error handling and debugging purposes.

# 5. `api_boundary` & `wraps` together

**Minimal example:**

```python
# 1. Definition
@api_boundary
def transform(f):
    @wrap(f)
    @api_boundary
    def impl(*args, **kwargs):
        ...
    return impl  

def my_func(*args, **kwargs):
    ...

# 2. Transform the custom function
transformed_func = transform(my_func)

# 3. Call the transformed function
transformed_func(...)
```

1. **Definition**: The `transform` function is wrapped by `api_boundary` ==> anytime `transform` is called, it is called inside a try-catch statement.
2. **Transform**: `transform` is called with argument is `my_func`. It take return inner `impl` function (which contains reference to `my_func` and can call `my_func` to do work). The `impl` function is also wrapped by `api_boundary` so every time it is called, it is called inside a try-catch statement. The `impl` function is also wrapped inside `wraps` which change its metadata like documentation, signature, name, path, ...
3. **Call the transformed**: When the transformed is called, first it is put inside a try-catch statement due to the wrapper `api_boundary`. Then the body of `impl` function is called.

Some variants of this pattern:

```python
@partial(api_boundary, repro_api_name="foobar")
def transform(f):
    ...
```

==> Same thing as above but the `api_boundary` wrapper has 1 more keyword argument. Whereas in the above example, it must use default argument.

# 6. Function `transformation2`

```python
@curry
def transformation2(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  """Adds one more transformation to a WrappedFun.

  Args:
    gen: the transformation generator function
    fun: a WrappedFun on which to apply the transformation
    gen_static_args: static args for the generator function
  """
  return fun.wrap(gen, gen_static_args, None)
```

In expanded form:

`tranformation2 = Partial(partial, transformation2_)` in which `transformation2_` is the originally defined function.

Uses of `tranformations`:

```python
@transformation2
def foobar(f: Callable, ...):
    ...
```

In expanded form:

```
foobar = Partial(tranformation2_, foobar_)
```

When `foobar` is called with `result = foobar(fun, *static_args)`, it expand to:

```
result = transformation2_(foobar_, fun: WrappedFun, *static_args)
       = fun.wrap(foobar_, *static_args) # type: WrappedFun
```

In short: a function wrapped by `transformation2` does not called with its own signature. Instead the signature is `(WrappedFunc, tuple[`
