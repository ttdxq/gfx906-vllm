# vLLM for AMD gfx906

[![vLLM](https://img.shields.io/badge/vLLM-gfx906-red)](https://github.com/vllm-project/vllm)
[![ROCm](https://img.shields.io/badge/ROCm-6.3+-purple)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/vllm-project/vllm/blob/main/LICENSE)

**专为 AMD gfx906 GPU（Radeon VII、Radeon Pro VII、Instinct MI50/MI60）优化的 LLM 推理引擎**

## 版权声明

本项目基于 [vLLM](https://github.com/vllm-project/vllm) 开发，感谢原项目所有贡献者。

- **原始项目**: [vLLM](https://github.com/vllm-project/vllm) by UC Berkeley Sky Computing Lab
- **许可证**: Apache License 2.0
- **本项目**: 包含针对AMD gfx906 GPU的优化和修复
- **衍生与参考来源**: 本分支在遵循 Apache-2.0 的前提下，吸收并参考了社区已有的 ROCm/gfx906 适配工作，相关归属见 [NOTICE](NOTICE)

详见 [NOTICE](NOTICE) 文件了解完整的归属声明。

## 项目状态

⚠️ **本项目由社区维护中。** 原作者已归档其工作，但本分支持续更新以保持与最新 vLLM 版本的兼容性。

## 最新更新

### ✅ 2026年6月更新

- **Qwen3.5 GGUF reasoning parser** - 新增并自动接入 `qwen3_5` reasoning parser，避免误用其他模型的 parser。
- **Qwen3.5 GGUF thinking scaffold 修复** - 修正 chat prompt 中 `<think>` / `</think>` 处理，兼容 Qwen3.5 默认输出行为。
- **Qwen3.5 GGUF dense fallback 修复** - 修复 GGUF dense 权重路径在 gfx906 上的兼容问题，避免错误反量化路径影响推理。
- **Qwen3.5 GDN projection 合并优化** - 将 Gated DeltaNet 的多组 linear 合并，减少 GGUF 路径上的额外算子开销。
- **Qwen3.5 GGUF piecewise compile 支持** - 补齐 layernorm、mRoPE、KV cache reshape 等 gfx906 capture-safe fallback，支持 `PIECEWISE` CUDA/HIP Graph 路径。
- **Qwen3.5 GGUF FULL decode graph 支持** - 补齐 causal conv1d、fused recurrent、sigmoid gating、unified attention 等 decode fallback，已验证 `FULL_AND_PIECEWISE` 可完成 FULL decode graph capture。
- **当前推荐配置** - Qwen3.5 GGUF 在 gfx906 上建议使用 `--reasoning-parser qwen3_5`，并优先使用 `FULL_AND_PIECEWISE` + `max_cudagraph_capture_size=128`。
- **当前限制** - 默认 capture size 512 仍可能因显存压力 OOM；27B 级 GGUF 仍属实验支持，复杂长输出质量需继续测试。

### 支持的模型

**⚠️ 重要提示：模型支持状态**

本项目已从上游 vLLM 同步了大量新模型文件（49+ 个模型），但这些模型**未经 ROCm/gfx906 架构的适配和测试**。虽然文件已包含在代码库中，但不保证在 AMD gfx906 GPU 上能正常运行。

**已测试并确认可用：**
- ✅ Qwen/Qwen3.5-0.8B（非多模态）
- ✅ Qwen/Qwen3.5-2B（非多模态）
- ✅ Qwen/Qwen3.5-4B（非多模态）
- ✅ Qwen/Qwen3.5-9B（非多模态）

**已知无法运行：**
- ❌ **Qwen/Qwen3.5-35B-A3B-GPTQ-Int4** - MoE + GPTQ Int4 量化组合存在兼容性问题，导致服务无法正常启动或推理失败
- ❌ **cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit** - 在 ROCm 6.4 + gfx906 上实测仍无法正常运行，现象包括返回整串 `!` 或在引擎初始化/推理阶段崩溃

**未经测试的模型（同步自上游，可能不工作）：**
- 🔄 其他 MoE 模型（exaone_moe, glm4_moe_lite 等）
- 🔄 视觉模型（colqwen3, molmo2 等）
- 🔄 音频模型（whisper, funasr 等）
- 🔄 嵌入模型（colbert, voyage 等）
- 🔄 其他新模型

如需使用这些模型，建议先测试基本功能。遇到问题请提交 issue。

## 项目简介

这是 [vLLM](https://github.com/vllm-project/vllm) 的修改版本，专门用于 AMD gfx906 系列 GPU。它包含了针对 gfx906 架构的 ROCm 兼容性优化和变通方案。

**原作者**: [nalanzeyu](https://github.com/nalanzeyu/vllm-gfx906)

## 系统要求

- **硬件**: AMD gfx906 GPU（Radeon VII、Radeon Pro VII、Instinct MI50、Instinct MI60）
- **ROCm**: 6.3+（需要内核模式驱动）
- **Python**: 3.10+
- **Triton**: triton-gfx906 v3.5.0+gfx906（[安装指南](https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906)）
- **操作系统**: Linux（在 Ubuntu 上测试）

## 安装方式

### 从源码构建

```bash
# 安装系统依赖
sudo apt install python3-venv python3-dev

# 克隆仓库
git clone https://github.com/ttdxq/gfx906-vllm.git
cd vllm-gfx906

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 ROCm 版本的 PyTorch
pip install torch==2.9 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# 安装依赖
pip install -r requirements/rocm-build.txt -r requirements/rocm.txt

# 安装 vLLM
pip install --no-build-isolation -e .
```

## 使用方法

### 启动服务

```bash
# 对于多模态模型（禁用多模态功能）
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3_5 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'

# 指定 GPU 内存使用率
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3_5
```

### Qwen3.5-27B GGUF（gfx906 实验状态）

对于 `Qwen3.5-27B` 级别的 GGUF，当前仓库仍以**原生 vLLM 路径排障**为主。现阶段建议仅将其视为实验性支持：

```bash
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=1
export HSA_ENABLE_SDMA=0
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_COMPILATION_MODE=0

vllm serve /root/model/Qwen3.5-27B-Q6_K.gguf \
  --tokenizer /root/model/Qwen3.5-27B-UD-Q6_K_XL-repo \
  --tokenizer-mode auto \
  --trust-remote-code \
  --port 8001 \
  --tensor-parallel-size 1 \
  --kv-cache-memory-bytes 268435456 \
  --max-model-len 256 \
  --reasoning-parser qwen3_5 \
  --compilation-config '{"mode":3,"backend":"eager","cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":128}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

**当前观察：**

- ✅ 服务启动与基础 OpenAI 接口链路可打通
- ✅ 简单问答、数字题、部分短回答已明显改善
- ✅ `FULL_AND_PIECEWISE` + `max_cudagraph_capture_size=128` 已可完成 FULL decode graph capture
- ✅ reasoning 内容会进入 OpenAI 响应的 `reasoning` / `reasoning_content` 字段，普通 `content` 可能为空
- ⚠️ 默认 capture size 512 仍可能因显存压力 OOM
- ⚠️ 复杂长回答质量仍需继续测试
- ⚠️ 目前仅建议作为实验性验证，不建议视为稳定生产支持

### 测试 API

```bash
# 检查模型可用性
curl http://localhost:8000/v1/models

# 发送对话请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "你好！请介绍一下你自己"}],
    "max_tokens": 100
  }'
```

### Python 客户端示例

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-required"
)

# 发送对话请求
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "user", "content": "写一首关于编程的俳句"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 核心修改

### 1. ROCm GEMM 优化
- **问题**: `torch.nn.functional.linear` 在 gfx906 上触发 `HIPBLAS_STATUS_INTERNAL_ERROR`
- **解决方案**: 使用通用 Triton 矩阵乘法实现，避免 hipBLAS 调用
- **修改文件**: `vllm/model_executor/layers/utils.py`

### 2. CacheConfig 兼容性
- 修复混合模型的 `mamba_cache_mode` 属性问题
- 更新 Qwen3.5 模型加载逻辑

### 3. 平台检测
- 改进 gfx906 的 ROCm 平台检测
- 优化注意力后端选择机制

## 量化支持

基于原项目测试结果：
- ✅ **GPTQ** - 推荐
- ✅ **AWQ** - 推荐
- ✅ **W4A16 INT** - 支持（通过 llm-compressor）
- ⚠️ **MoE 量化模型** - 速度显著较慢，不推荐
- ⚠️ **非量化模型** - 略慢，但可用

gfx906 上 AWQ 默认继续使用当前较保守的 Triton 路径。如需试验 `vllm-gfx906-mobydick` 风格的 GPTQ-compatible AWQ 路径，可显式设置：

```bash
export VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ=1
```

该开关默认关闭，仅建议在排查 Triton AWQ 兼容性或对比两条 AWQ 路径行为时使用。

详细信息请参阅 [Issue #29](https://github.com/nlzy/vllm-gfx906/issues/29)。

## 已知限制

1. **不支持多模态** - 必须使用 `--limit-mm-per-prompt` 禁用
2. **首次推理较慢** - Triton 内核首次运行时需要编译
3. **内存占用较大** - KV cache 预分配会占用大量 GPU 内存
4. **实验性质** - 使用风险自负
5. **MoE + GPTQ 量化兼容性** - Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 等大型 MoE + GPTQ Int4 量化模型存在已知问题：
   - Triton kernel 在处理特定分块大小时会出现内存访问错误
   - aiter 后端在复杂 MoE 路由场景下不稳定
   - 建议使用非量化 MoE 模型或较小的 MoE + GPTQ 模型
6. **Qwen3.5 GGUF on gfx906** - 当前 27B 级 GGUF 在原生 vLLM 路径下仍可能出现复杂提示上的题目复述、thinking 文本异常或语义漂移，暂不建议视为稳定支持

## 性能优化建议

1. **减小 `max-model-len`** - 对于小模型可以节省内存
2. **使用 `--gpu-memory-utilization`** - 显式管理内存使用
3. **优先使用 `FULL_AND_PIECEWISE`** - Qwen3.5 GGUF 推荐配合 `max_cudagraph_capture_size=128`
4. **重启前清理进程** - 使用 `pkill -9 -f "vllm serve"` 杀死现有进程

## 故障排除

### 服务在首次请求时挂起
这是正常现象 - Triton 正在编译内核。请等待 1-2 分钟。

### GPU 内存错误
- 杀死现有进程: `pkill -9 -f "vllm serve"`
- 降低 GPU 内存使用率: `--gpu-memory-utilization 0.7`

### HIPBLAS 错误
最新版本应该已修复。如果遇到，请提交 issue。

## 贡献

欢迎贡献！你可以：
- 报告 bug
- 提出新功能建议
- 提交 pull request
- 改进文档

## 致谢

- **原始 vLLM**: [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu)
- **ROCm 移植**: [Said-Akbar/vllm-rocm](https://github.com/Said-Akbar/vllm-rocm)
- **gfx906 分支**: [nalanzeyu/vllm-gfx906](https://github.com/nalanzeyu/vllm-gfx906)
- **gfx906 社区适配分支**: [ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick)
- **Triton for gfx906**: [nlzy/triton-gfx906](https://github.com/nlzy/triton-gfx906)

## 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

## 引用

如果你在研究中使用了本分支，请同时引用原始 vLLM 论文并说明 gfx906 适配：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/ttdxq/gfx906-vllm/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/ttdxq/gfx906-vllm/discussions)

---

**注意**: 本项目为社区维护的分支。使用风险自负，特别是作为硬件购买的参考依据。
