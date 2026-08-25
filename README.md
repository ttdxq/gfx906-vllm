[English](./README_EN.md)

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

### ✅ 2026年8月更新

- **Qwen3.5 多模态 GGUF 支持** - 补齐 Qwen3.5 GGUF 主模型与 `mmproj` 的视觉配置、权重名称映射、Conv3d patch embedding 合并和多模态输入处理链路。

### 支持的模型

**⚠️ 重要提示：模型支持状态**

本项目已从上游 vLLM 同步了大量新模型文件（49+ 个模型），但这些模型**未经 ROCm/gfx906 架构的适配和测试**。虽然文件已包含在代码库中，但不保证在 AMD gfx906 GPU 上能正常运行。

**已测试并确认可用：**

- ✅ Qwen/Qwen3.5-0.8B(原始F16权重)
- ✅ Qwen/Qwen3.5-2B(原始F16权重)
- ✅ Qwen/Qwen3.5-4B(原始F16权重)
- ✅ Qwen/Qwen3.5-9B(原始F16权重)
- ✅ unsloth/Qwen3.5-27B-GGUF
- ✅ unsloth/Qwen3.6-27B-GGUF
- ✅ unsloth/Qwen3.8-27B-GGUF

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
- **ROCm**: 支持以下社区兼容组合（均需要内核模式驱动）
  - ROCm 6.3 + PyTorch 2.9 + triton-gfx906 v3.5.0+gfx906（原有兼容组合）
  - ROCm 7.2 + PyTorch 2.11 + triton-rocm 3.6.0（当前仓库依赖组合）
- **Python**: 3.10+
- **操作系统**: Linux（在 Ubuntu 上测试）

ROCm 6.3 组合的 Triton 安装方式请参阅
[triton-gfx906 安装指南](https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906)。
当前仓库的依赖文件默认使用 ROCm 7.2 组合。该组合需要可用于 gfx906 的
rocBLAS Tensile 文件；必要时通过 `ROCBLAS_TENSILE_LIBPATH` 指定其路径。

## 安装方式

### 从源码构建

```bash
# 安装系统依赖
sudo apt install python3-venv python3-dev

# 克隆仓库
git clone https://github.com/ttdxq/gfx906-vllm.git
cd gfx906-vllm

# 创建虚拟环境
uv venv --python 3.12 .venv
source .venv/bin/activate

# 安装当前 ROCm 7.2 构建和运行依赖
uv pip install -r requirements/rocm-build.txt -r requirements/rocm.txt

# 构建并安装 vLLM
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  uv pip install --no-build-isolation -e .
```

上述源码构建命令对应当前仓库的 ROCm 7.2 + PyTorch 2.11 依赖。继续使用
ROCm 6.3 原有组合时，请按 triton-gfx906 安装指南准备对应的 PyTorch 和
Triton 环境，不要混用两套 ROCm/PyTorch/Triton 依赖。

## 使用方法

### 启动服务

```bash
# 文本模型或仅使用文本功能时，可以显式禁用多模态输入
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

### 双卡张量并行（TP=2）

> 前置条件：`--tensor-parallel-size 2` 需要替换 RCCL。AMD 官方 RCCL（含 ROCm 7.2 官方仓版本）
> 不包含 gfx906 设备内核，直接启动会在初始化阶段以 `invalid kernel file` /
> `invalid device function` 等错误失败。需安装 gfx906 构建的 RCCL 替换库（与官方
> 2.27.7 同源，仅构建目标为 gfx906），适用于 torch `+rocm7.2` 轮子。

**1. 安装 gfx906 RCCL**（从 Releases 下载 `rccl-gfx906-2.27.7-rocm7.2.4.tar.gz` 解压后）：

```bash
# conda / venv 环境（先激活）
conda activate <你的环境>
./install.sh
HIP_VISIBLE_DEVICES=0,1 python rccl_probe.py
# 预期输出 ALLREDUCE_OK: [2.0, 2.0, 2.0, 2.0] 即安装成功；回滚用 ./install.sh --rollback

# uv 项目环境（--no-sync 必须带，防止隐式 sync 破坏定制 torch）
uv run --no-sync ./install.sh
```

若无预编译包，可自行构建（约 25 分钟）：

```bash
git clone --depth 1 --branch rocm-7.2.4 https://github.com/ROCm/rccl
cd rccl && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS="gfx906" -G Ninja
ninja -j$(nproc)
# 用 build/librccl.so.1.0 按上述 install.sh 同样方式替换 torch/lib 内的 librccl
```

**2. 启动 TP=2 服务**：

```bash
HIP_VISIBLE_DEVICES=0,1 \
vllm serve /path/to/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --reasoning-parser qwen3_5 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

说明：

- 单卡（TP=1）不经过 RCCL，无需上述替换，不受任何影响。
- 双卡下贪心输出与单卡可能在个别"近平局"token 上分岔（浮点求和顺序差异），
  属正常现象，不是安装或权重问题。

### Qwen3.5系列 多模态 GGUF支持

主模型和对应 `mmproj` 应放在同一目录，并保持可识别的配对命名。

为约束启动 profiling 使用的 dummy 图片尺寸，建议设置与业务图片上限相符的 `width` 和 `height`：

```bash
--limit-mm-per-prompt '{"image":{"count":1,"width":512,"height":512},"video":0}'
```

`width` 和 `height` 用于约束启动 profiling 的 dummy 图片尺寸，并不是运行时图片的硬限制。部署时应使用与实际最大图片尺寸匹配的 profiling 配置。

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
- **问题**: gfx906 上不同 batch、矩阵形状和 dtype 对 ROCm GEMM 路径的兼容性与性能差异较大
- **解决方案**: 按输入形状选择实现；符合条件的单 token FP16 路径使用 `LLMM1`，部分非 gfx9 小 batch 使用 Triton，其余 gfx906 路径使用 PyTorch/ROCm GEMM fallback
- **修改文件**: `vllm/model_executor/layers/utils.py`

### 2. CacheConfig 兼容性
- 修复混合模型的 `mamba_cache_mode` 属性问题
- 更新 Qwen3.5 模型加载逻辑

### 3. 平台检测
- 改进 gfx906 的 ROCm 平台检测
- 优化注意力后端选择机制

## 量化支持

基于原项目测试结果：
- ⚠️ **GPTQ** - 部分模型可用，但 kernel、张量形状和 MoE 组合仍可能存在兼容性或性能问题
- ⚠️ **AWQ** - 部分 dense 模型可用；默认使用 Triton 路径，部分模型仍可能输出异常或启动失败
- ⚠️ **W4A16 INT** - 部分模型不可用（通过 llm-compressor）
- ⚠️ **MoE 量化模型** - 速度显著较慢，不推荐
- ⚠️ **非量化模型** - 略慢，但可用

gfx906 上 AWQ 默认继续使用当前较保守的 Triton 路径。如需试验 `vllm-gfx906-mobydick` 风格的 GPTQ-compatible AWQ 路径，可显式设置：

```bash
export VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ=1
```

该开关默认关闭，仅建议在排查 Triton AWQ 兼容性或对比两条 AWQ 路径行为时使用。

详细信息请参阅 [Issue #29](https://github.com/nlzy/vllm-gfx906/issues/29)。

## 已知限制

1. **首次推理较慢** - Triton 内核首次运行时需要编译
2. **内存占用较大** - KV cache 预分配会占用大量 GPU 内存
3. **实验性质** - 使用风险自负
4. **MoE + GPTQ 量化兼容性** - Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 等大型 MoE + GPTQ Int4 量化模型存在已知问题：
   - Triton kernel 在处理特定分块大小时会出现内存访问错误
   - aiter 后端在复杂 MoE 路由场景下不稳定
   - 建议使用非量化 MoE 模型或较小的 MoE + GPTQ 模型

## 性能优化建议

1. **减小 `max-model-len`** - 对于小模型可以节省内存
2. **使用 `--gpu-memory-utilization`** - 显式管理内存使用
3. **保持默认 eager decode** - Qwen3.5/Qwen3.8 GGUF 的 CUDA/HIP Graph 路径仅用于排障和实验验证
4. **限制多模态 profiling 尺寸** - 通过 `--limit-mm-per-prompt` 设置与业务相符的图片数量和最大测试尺寸
5. **重启前确认进程归属** - 使用 `rocm-smi` 和 `ps` 检查占用，避免终止其他 GPU 工作线的进程

## 故障排除

### 服务在首次请求时挂起
这是正常现象 - Triton 正在编译内核。请等待 1-2 分钟。

### GPU 内存错误
- 使用 `rocm-smi` 检查目标 GPU 的显存占用和进程归属
- 降低 `--max-model-len`、`--max-num-batched-tokens` 或 KV cache 大小
- 多模态模型应通过 `--limit-mm-per-prompt` 降低 profiling 图片尺寸
- 如果 OOM 发生在启动阶段的视觉 profiling，可临时使用 `--skip-mm-profiling` 判断是否为 profiling 峰值导致

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
