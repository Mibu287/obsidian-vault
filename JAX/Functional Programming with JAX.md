
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