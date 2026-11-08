// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/custom_class.h>
#include <hip/hip_runtime.h>

// A host-side sequence of existing Triton kernels and the existing top-k op.
// This neither captures a GPU graph nor changes a kernel's arithmetic.
class Qwen4ExpQSADecodePlan : public torch::CustomClassHolder {
  struct Step {
    int64_t function = 0;
    std::vector<int64_t> grid, values, bindings;
    int64_t threads = 0, shared = 0;
    std::vector<int64_t> topk;
  };
  int64_t device_;
  std::vector<Step> steps_;

 public:
  explicit Qwen4ExpQSADecodePlan(int64_t device) : device_(device) {}

  void add_kernel(int64_t function, std::vector<int64_t> grid,
                  int64_t threads, int64_t shared,
                  std::vector<int64_t> values,
                  std::vector<int64_t> bindings) {
    TORCH_CHECK(function && grid.size() == 3 && threads > 0 &&
                    threads <= 1024 && shared >= 0 &&
                    bindings.size() % 3 == 0,
                "Invalid QSA kernel launch descriptor");
    for (auto size : grid) TORCH_CHECK(size > 0, "Empty QSA launch grid");
    for (size_t i = 0; i < bindings.size(); i += 3) {
      TORCH_CHECK(bindings[i] >= 0 && bindings[i] < values.size() &&
                      bindings[i + 1] >= 0,
                  "Invalid QSA tensor argument binding");
    }
    Step step;
    step.function = function;
    step.grid = std::move(grid);
    step.threads = threads;
    step.shared = shared;
    step.values = std::move(values);
    step.bindings = std::move(bindings);
    steps_.push_back(std::move(step));
  }

  void add_topk(std::vector<int64_t> indices) {
    TORCH_CHECK(indices.size() == 4, "Expected QSA top-k tensor slots and k");
    Step step;
    step.topk = std::move(indices);
    steps_.push_back(std::move(step));
  }

  void run(std::vector<at::Tensor> tensors) {
    TORCH_CHECK(!tensors.empty(), "QSA plan needs tensor operands");
    const c10::DeviceGuard guard(tensors[0].device());
    for (const auto& tensor : tensors) {
      TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device_,
                  "QSA plan operands must stay on their compilation device");
    }
    const auto stream = c10::hip::getCurrentHIPStream();
    for (const auto& step : steps_) {
      if (!step.topk.empty()) {
        static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "_C::top_k_per_row_decode", "");
        for (size_t i = 0; i < 3; ++i) {
          TORCH_CHECK(step.topk[i] >= 0 && step.topk[i] < tensors.size(),
                      "QSA top-k operand index is out of range");
        }
        const auto& logits = tensors[step.topk[0]];
        c10::Stack stack{
            logits, int64_t(1), tensors[step.topk[1]], tensors[step.topk[2]],
            logits.size(0), logits.stride(0), logits.stride(1), step.topk[3]};
        op.callBoxed(&stack);
        continue;
      }
      auto values = step.values;
      for (size_t i = 0; i < step.bindings.size(); i += 3) {
        auto slot = step.bindings[i + 1];
        TORCH_CHECK(slot < tensors.size(), "QSA operand index is out of range");
        values[step.bindings[i]] =
            reinterpret_cast<int64_t>(tensors[slot].data_ptr()) +
            step.bindings[i + 2];
      }
      std::vector<void*> args;
      args.reserve(values.size());
      for (auto& value : values) args.push_back(&value);
      auto status = hipModuleLaunchKernel(
          reinterpret_cast<hipFunction_t>(step.function),
          step.grid[0], step.grid[1], step.grid[2],
          step.threads, 1, 1, step.shared, stream.stream(),
          args.data(), nullptr);
      TORCH_CHECK(status == hipSuccess, "QSA kernel launch failed: ",
                  hipGetErrorString(status));
    }
  }
};
