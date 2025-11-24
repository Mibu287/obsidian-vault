# 1. Node

```C++
// file: torch/csrc/autograd/function.h

struct TORCH_API Node : std::enable_shared_from_this<Node> {
  // Sequence number used to correlate backward nodes with forward ops in the
  // profiler and provide determinism in the engine.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  uint64_t sequence_nr_;

  // See NOTE [ Topological Number ]
  uint64_t topological_nr_ = 0;

  // Tracks whether this node has been added as the next_edge of another node
  // via set_next_edge(s), which always calls topological_nr() of all its
  // children See NOTE [ Topological Number ] for why we need this.
  mutable bool has_parent_ = false;

  // Id of the thread that created the instance
  uint64_t thread_id_ = 0;

  // Note [Thread Safety on Autograd Node]
  std::mutex mutex_;

  edge_list next_edges_;
  PyObject* pyobj_ = nullptr; // weak reference
  std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;

  // NOTE [Hooks ordering]
  // We have 3 separate fields for pre hooks registered to the autograd nodes
  // because the conditions under which they execute are different, and we
  // want more fine-grained control over the order in which different types
  // of hooks are executed.
  // - pre_hooks  are only executed when the node itself is executed
  // - tensor_pre_hook is executed as long as the engine traverses over it
  //   even if that node won't be executed.
  // - retains_grad_hook are like tensor_pre_hooks except they are always
  //   ordered after all other tensor pre hooks
  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;
  std::vector<std::unique_ptr<FunctionPreHook>> tensor_pre_hooks_;
  std::unordered_map<size_t, std::unique_ptr<FunctionPreHook>>
      retains_grad_hooks_;
  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;
  at::SmallVector<InputMetadata, 2> input_metadata_;
```

Each `Node` hold a vector to `Edge`s (`next_edges`)

# 2. Edge

```C++
// file: torch/csrc/autograd/function.h

struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  /// Convenience method to test if an edge is valid.
  bool is_valid() const noexcept {
    return function != nullptr;
  }

  // Required for use in associative containers.
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  std::shared_ptr<Node> function;

  /// The identifier of a particular input to the function.
  uint32_t input_nr;
};
```

An `Edge` contain a shared pointer to a `Node` and also keep record of the input identifier.

# 3. Nodes and Edges in a system

A system of `Node`s and `Edge`s form a directed graph. In which, each `Node` is a vertex and each `Edge` is an out-going edge from such `Node`
Each `Node` is a function in the graph, each `Edge` is an input or an output of a function.