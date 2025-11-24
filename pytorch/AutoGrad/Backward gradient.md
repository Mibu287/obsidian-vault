
To calculate backward gradient, a Tensor call its `backward` method to kickoff the gradient calculation process. The method then call `torch::autograd::backward` which implement the autograd calculation logic.


```C++
// file: torch/csrc/autograd/autograd.cpp

namespace torch::autograd {
static variable_list run_backward(
    const variable_list& outputs,
    const variable_list& grad_outputs,
    bool keep_graph,
    bool create_graph,
    const variable_list& inputs,
    bool allow_unused,
    bool accumulate_grad) {
  size_t num_tensors = outputs.size();
  edge_list roots;
  roots.reserve(num_tensors);
  for (const auto i : c10::irange(num_tensors)) {
    const Variable& output = outputs[i];
    auto gradient_edge = impl::gradient_edge(output);
    TORCH_CHECK(
        gradient_edge.function,
        "element ",
        i,
        " of tensors does not require grad and does not have a grad_fn");
    roots.push_back(std::move(gradient_edge));
  }

  edge_list output_edges;
  if (!inputs.empty()) {
    size_t num_inputs = inputs.size();
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      const Variable& input = inputs[i];
      const auto output_nr = input.output_nr();
      auto grad_fn = input.grad_fn();
      if (!grad_fn) {
        grad_fn = impl::try_get_grad_accumulator(input);
      }
      if (accumulate_grad) {
        input.retain_grad();
      }
      TORCH_CHECK(
          input.requires_grad(),
          "element ",
          i,
          " of the input tensors does not require grad");
      if (!grad_fn) {
        // See NOTE [ Autograd Unreachable Input ] for details
        output_edges.emplace_back(std::make_shared<Identity>(), 0);
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }
  variable_list grad_inputs = Engine::get_default_engine().execute(
      roots,
      grad_outputs,
      keep_graph,
      create_graph,
      accumulate_grad,
      output_edges);
  // check if grad_inputs contains None or not base on the allow_unused flag
  if (!inputs.empty() && !allow_unused) {
    size_t num_inputs = inputs.size();
    for (const auto i : c10::irange(num_inputs)) {
      TORCH_CHECK(
          grad_inputs[i].defined(),
          "element ",
          i,
          "of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
    }
  }
  return grad_inputs;
}
```

Main logic of the function:

1. Create root edges which is gradient of the output variable
2. Create output edges which is grad function of input tensor. In simple case of calling `final_result.grad()`, the `inputs` parameter is empty ==> the `output_edges` variable is also empty. Why `inputs` parameter is needed in this function is unclear at this time. Maybe new grad result is accumulated into the input's grad? More investigation is needed.
3. The autograd engine is called to execute the graph.
4. The engine will call `apply` method of each edge to calculate gradient of the result variable with regard to each element in the compute graph
5. The gradient result of each node is saved into to each corresponding tensor