#JAXML #DeepDive #DeepLearning

# 1.  class `Traced`

```python
class Traced(Stage):
  """Traced form of a function specialized to argument types and values.

  A traced computation is ready for lowering. This class carries the
  traced representation with the remaining information needed to later
  lower, compile, and execute it.
  """
  __slots__ = ["jaxpr", "args_info", "fun_name", "_out_tree", "_lower_callable",
               "_args_flat", "_arg_names", "_num_consts"]

  def __init__(self, jaxpr: core.ClosedJaxpr, args_info, fun_name, out_tree,
               lower_callable, args_flat=None, arg_names=None,
               num_consts: int = 0):
    self.jaxpr = jaxpr
    self.args_info = args_info
    self.fun_name = fun_name
    self._out_tree = out_tree
    self._lower_callable = lower_callable
    self._args_flat = args_flat
    self._arg_names = arg_names
    self._num_consts = num_consts
```

```python
class Stage:
  args_info: Any  # PyTree of ArgInfo
```


# 2. class `Trace`

class `Trace` contains logic about how to handle primitives (operations, functions) and tracers (i.e. operands).

```python
TracerType = TypeVar('TracerType', bound='Tracer')

class Trace(Generic[TracerType]):
  __slots__ = ("__weakref__", "_invalidated", "_weakref")

  def __init__(self):
    self._invalidated = False
    # We frequently need a weakref to a trace, so let's precompute one.
    self._weakref = weakref.ref(self)

  def process_primitive(self, primitive, tracers, params):
    ...
    
  def invalidate(self):
    ...

  def is_valid(self):
    ... 

  def __repr__(self):
    ...
    
  def process_call(self, call_primitive, f, tracers, params):
    ...

  def process_map(self, map_primitive, f, tracers, params):
    ...
```

```python
class JaxprTrace(Trace['JaxprTracer']):
  def __init__(self, parent_trace:Trace, name_stack: source_info_util.NameStack, tag:TraceTag):
    super().__init__()
    self.name_stack = name_stack
    self.tag = tag
    self.parent_trace = parent_trace


  def process_primitive(self, primitive, tracers, params):
    with core.set_current_trace(self.parent_trace):
      if primitive in custom_partial_eval_rules:
        tracers = map(self.to_jaxpr_tracer, tracers)
        return custom_partial_eval_rules[primitive](self, *tracers, **params)
      else:
        return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    # By default, if all the input tracers are known, then bind the primitive
    # and consider all outputs known. Otherwise, stage the application into the
    # jaxpr and consider all outputs unknown.
    tracers = map(self.to_jaxpr_tracer, tracers)
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind_with_trace(self.parent_trace, consts, params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval, effects = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(tracers, out_tracers, primitive, params, effects, source)
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      out_tracer.recipe = new_eqn_recipe(tracers, [out_tracer], primitive,
                                         params, effects, source)
      return out_tracer
```

# 2. class `Tracer`

```python
class Tracer(typing.Array, metaclass=StrictABCMeta):
  __array_priority__ = 1000
  __slots__ = ['_trace', '_line_info']
  __hash__ = None  # type: ignore

  _trace: Trace
  _line_info: source_info_util.SourceInfo | None

  dtype = _aval_property('dtype')
  ndim = _aval_property('ndim')
  size = _aval_property('size')
  shape = _aval_property('shape')
```

```python
class DynamicJaxprTracer(core.Tracer):
  __slots__ = ['aval', '_debug_info']

  def __init__(self, trace: DynamicJaxprTrace,
               aval: core.AbstractValue,
               line_info: source_info_util.SourceInfo | None = None):
    self._trace = trace
    self._line_info = line_info
    self._debug_info = self._trace.frame.debug_info  # for UnexpectedTracerError
    self.aval = aval  # type: ignore[misc]
```

# 4. Logic

Example:

```python
def f(x):
    return jax.lax.add(jax.lax.sin(x), jax.lax.cos(x))

x = ...

# Step 1: Transform the function
g = jax.make_jaxpr(
    jax.vmap(
        jax.vmap(f),
    ),
)

# Step 2: Call the function
g(x)
```

- At step 1, the function is transformed. No work is done yet. JAX utilize function decorator in Python to transform functions.
- At step 2, the function is called. First, the function is executed in 2 phases:
    - Phase 1: The function g is called, it make trace of `vmap` its trace parent. Also, it wraps input tracers inside its own Tracer class. Then it recursively call inner function (`vmap`). At the end of this phase, JAX build a linked list for trace and nested (i.e. multi-layered) tracers.
    - Phase 2: `f` is called with argument `x` is a nested tracer. The object trace contains logic to handle deal with the input tracers and return output tracers. Then, it unwrap output tracers, change trace object with its trace parent and pass the output tracers as input in trace parent methods.

In short, `g` prepare inputs and recursively call its inner functions. Step 1: input go from outermost to innermost. At each layer, more information is added to the input (i.e. tracer is wrapped with another Tracer class at each layer). Step 2: Function called, inputs are handled by logic of each layer. Control flow go from innermost to outermost. At each layer, information specific at this layer is dropped (i.e. tracer object is dropped, only its inner tracer is passed to the outer layer).

