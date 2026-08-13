# GOAL: Decoupled Collaboration Training - 解耦工作层与协作层的迭代训练实验

## Meta Information

```yaml
goal_id: decoupled-collab-v1
status: NOT_STARTED  # NOT_STARTED | IN_PROGRESS | PAUSED | COMPLETED | FAILED
current_phase: null
current_step: null
last_checkpoint: null
created: 2026-08-14
hardware: 4×V100-32G (SSH direct)  # Volta sm_70; FP16 only (no bf16)
base_model: Qwen/Qwen3-4B (thinking mode)
eval_api: deepseek-v4-flash (via OpenAI-compatible API)
benchmark: MBPP+ (EvalPlus) + LiveCodeBench-easy
project_dir: /path/to/decoupled_collab  # 修改为实际服务器路径
# Recommended software stack for Qwen3-4B + V100-32G (see Step 0.2):
software_stack:
  python: "3.11"
  cuda_runtime: "12.1 or 12.4 (driver >= that)"
  torch: "2.5.1+cu121 or 2.6.0+cu124"   # FP16; avoid bf16
  transformers: ">=4.51.0,<4.53"         # Qwen3 requires >=4.51 (else KeyError: qwen3)
  vllm: "0.8.5"                         # Qwen3 needs >=0.8.5; V100 last common pip pin with sm_70
  trl: "0.15.2"                         # GRPO; needs transformers>=4.46
  peft: "0.14.0"
  accelerate: "1.2.1"
  inference_default: "huggingface+peft" # vLLM optional; adapter dirs need merge first
```

## Goal Summary

验证核心假设：通过 RL→分离→重生成→DPO 的迭代框架，可以在不损害模型coding能力的前提下，显著提升其面向人类的协作可读性。

**核心方法论**：利用 Qwen3 的 thinking mode 天然分离工作层（`<think>` block）和协作层（response text），无需复杂的token分类算法。

## Success Criteria

- [ ] Phase 1 完成后：Model_RL 在 MBPP+ 上 pass@1 比 Base 提升 ≥5%
- [ ] Phase 1 完成后：Model_RL 的可读性得分比 Base 下降（证明RL损害可读性）
- [ ] Phase 3 完成后：Model_Final 的可读性得分恢复到 ≥ Base 水平
- [ ] Phase 3 完成后：Model_Final 的 MBPP+ pass@1 与 Model_RL 差距 ≤2%
- [ ] LiveCodeBench-easy 上趋势与 MBPP+ 一致（排除contamination）
- [ ] 至少完成 2 个 cycle 的迭代，并记录收敛趋势

---

## Benchmark选择理由

### 主benchmark: MBPP+ (EvalPlus)

- **论文**: EvalPlus (NeurIPS 2023), 增强版测试用例
- **地址**: https://github.com/evalplus/evalplus
- **问题数**: ~500 Python编程题（sanitized split: 399题）
- **Qwen3-4B基线**: ~74% pass@1
- **RL提升空间**: 预期可提升到 80-85%
- **优点**:
  - 单步代码生成，完美适配简化实验设计
  - 增强测试用例(80+ tests/problem) → reward signal可靠
  - 社区认可度高，reviewer熟悉
  - 问题简单实用（字符串/列表/基础算法），4B模型能理解
- **训练/测试划分**:
  - 训练: 使用MBPP full set (974题) 作为GRPO训练prompt
  - 测试: 使用MBPP+ sanitized (399题) 作为评估

### 辅助benchmark: LiveCodeBench-easy

- **论文**: LiveCodeBench (ICLR 2025)
- **地址**: https://livecodebench.github.io/
- **作用**: Contamination-free验证
- **Qwen3-4B基线**: easy split ~60-70%
- **用法**: 仅用于评估，不用于训练 → 证明generalization

### 不使用的 & 理由

| Benchmark | 不用的原因 |
|-----------|-----------|
| HumanEval | 4B模型已86.6%，空间太小 |
| SWE-bench | 需要多步agent能力，4B跑不动 |
| BigCodeBench | 需要复杂library调用，超出4B能力 |
| LiveCodeBench hard | 35%基线，RL难以提升 |

---

## 环境准备 (Phase 0)

### Step 0.1: 创建项目结构

```bash
mkdir -p decoupled_collab/{src,configs,data,checkpoints,logs,results}
cd decoupled_collab
```

目标目录结构：
```
decoupled_collab/
├── src/
│   ├── train_grpo.py          # GRPO训练脚本
│   ├── collect_traces.py      # 收集agent trace（利用thinking mode）
│   ├── regen_collaboration.py # 用base model重新生成协作文本
│   ├── filter_pairs.py        # 用DeepSeek API过滤低质量pair
│   ├── train_dpo.py           # DPO训练脚本
│   ├── evaluate.py            # 评估脚本 (benchmark + readability)
│   ├── run_pipeline.py        # 全流程自动化master脚本
│   └── utils/
│       ├── prompts.py         # 所有prompt模板
│       ├── metrics.py         # 评估指标计算
│       ├── code_executor.py   # 安全代码执行sandbox
│       └── api_judge.py       # DeepSeek V4 Flash 可读性打分
├── configs/
│   ├── grpo_config.yaml
│   ├── dpo_config.yaml
│   └── pipeline_config.yaml
├── data/
│   ├── mbpp_train.jsonl       # MBPP训练集 (GRPO用)
│   ├── mbpp_plus_test.jsonl   # MBPP+ 测试集 (评估用)
│   ├── lcb_easy.jsonl         # LiveCodeBench easy split (评估用)
│   ├── traces/                # 收集到的traces
│   └── dpo_pairs/             # DPO训练对
├── checkpoints/
│   ├── cycle_0/
│   │   ├── model_rl/
│   │   ├── model_rl_dpo/
│   │   └── eval_results.json
│   └── cycle_1/ ...
├── logs/
├── results/
├── requirements.txt
└── GOAL.md  # 本文件
```

### Step 0.2: 安装依赖（Qwen3-4B × V100-32G 兼容栈）

> **为何改 pin（相对初版 GOAL）**  
> - 官方 Qwen3 卡：`transformers<4.51.0` → `KeyError: 'qwen3'`；thinking 需 `enable_thinking`（见 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)）。  
> - Qwen 文档：部署用 `vllm>=0.8.5`。  
> - V100 = **sm_70**：vLLM **v0.9+** 预编译镜像普遍丢掉 CC&lt;8.0；社区/官方 legacy 指引把 **`v0.8.5`** 当作仍带 Volta kernel 的常用上限（见 [vLLM production-stack V100 tutorial](https://github.com/vllm-project/production-stack/blob/main/tutorials/25-v100-legacy-gpu-deployment.md)）。  
> - 因此 **Qwen3 + V100 的 pip 交集**：`transformers>=4.51` + **`vllm==0.8.5`**；训练主路径用 HF+PEFT（fp16），vLLM 仅作可选加速。  
> - V100 **无 bf16 Tensor Core** → 全程 `float16` / `fp16=true`，禁止默认 bf16。

```bash
conda create -n collab python=3.11 -y
conda activate collab

# --- CUDA PyTorch (pick ONE; both OK on V100 if driver matches) ---
# Option A (widely available):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
# Option B (closer to Qwen "torch>=2.6" advice):
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
#   --index-url https://download.pytorch.org/whl/cu124

# --- Qwen3 + training stack ---
pip install "transformers>=4.51.0,<4.53"
pip install trl==0.15.2
pip install peft==0.14.0
pip install accelerate==1.2.1
pip install datasets==3.2.0
pip install bitsandbytes==0.45.0   # optional; V100 上优先 LoRA fp16，不必强依赖 4bit
pip install deepspeed==0.15.4      # optional ZeRO; DDP+LoRA 通常够用
pip install openai evalplus wandb jsonlines rich pyyaml tqdm huggingface_hub

# --- Optional: vLLM for collect/regen (NOT required for GRPO/DPO) ---
# Pin 0.8.5 for Qwen3 + V100 sm_70. Do NOT casually upgrade to 0.9+.
pip install vllm==0.8.5
export VLLM_USE_V1=0                 # V1 engine assumes CC>=8.0
# Prefer: dtype=half, --enforce-eager on Volta if you hit CUDA-graph issues
```

或直接：

```bash
# from decoupled_collab/
bash scripts/setup_env.sh
```

**V100 / Qwen3 运行约束（写进操作手册）**

| 项 | 要求 |
|----|------|
| dtype | `float16` only（配置里 `torch_dtype: float16`, `fp16: true`, `bf16: false`） |
| thinking | `enable_thinking=True`；缺则 fail-fast，禁止静默关掉 |
| 训练并行 | `accelerate launch --num_processes 4`；GRPO `num_samples_per_prompt` 必须整除 `4 × per_device_batch_size`（默认 4） |
| 推理默认 | HF + PEFT（`inference.use_vllm: false`） |
| vLLM | 仅完整/merged 模型目录；LoRA adapter 目录禁止直喂；`VLLM_USE_V1=0` |
| 采样（thinking） | Qwen 建议 temperature≈0.6, top_p≈0.95（相对旧 GOAL 的 0.7/0.9 可按阶段调） |

**检查点**:
```bash
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"gpu{i}", torch.cuda.get_device_name(i), f"sm_{major}{minor}")
    assert (major, minor) == (7, 0), "Expected V100 sm_70; adjust stack if different GPUs"
print("transformers", transformers.__version__)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 51)
try:
    import vllm
    print("vllm", vllm.__version__)
except ImportError:
    print("vllm not installed (OK if using HF-only inference)")
PY
```
期望：`ngpu` 为 4，每张卡 `sm_70`，`transformers>=4.51`。

### Step 0.3: 下载模型和数据

```bash
# 下载Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir ./models/Qwen3-4B

# 下载MBPP数据
python -c "
from datasets import load_dataset
# MBPP full (训练用)
ds = load_dataset('google-research-datasets/mbpp', 'full')
ds.save_to_disk('./data/raw/mbpp_full')
# MBPP sanitized (评估用)
ds2 = load_dataset('google-research-datasets/mbpp', 'sanitized')
ds2.save_to_disk('./data/raw/mbpp_sanitized')
print(f'Train: {len(ds[\"train\"])} problems')
print(f'Test (sanitized): {len(ds2[\"test\"])} problems')
"

# 下载LiveCodeBench easy split
pip install livecodebench  # 如果有官方包
# 或手动从 https://github.com/LiveCodeBench/LiveCodeBench 获取
```

**检查点**: 模型目录和数据目录存在且非空

### Step 0.4: 准备训练数据格式

```python
# src/prepare_data.py
"""将MBPP转换为实验所需格式"""
import json
from datasets import load_from_disk

def prepare_grpo_tasks():
    """准备GRPO训练任务: prompt + test_cases"""
    ds = load_from_disk('./data/raw/mbpp_full')
    
    tasks = []
    for item in ds['train']:
        task = {
            "task_id": f"mbpp_{item['task_id']}",
            "prompt": item['text'],  # 自然语言描述
            "test_cases": item['test_list'],  # assert语句列表
            "code_solution": item['code'],  # ground truth (不给模型看)
        }
        tasks.append(task)
    
    with open('./data/mbpp_train.jsonl', 'w') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(tasks)} GRPO training tasks")

def prepare_eval_tasks():
    """准备评估任务"""
    ds = load_from_disk('./data/raw/mbpp_sanitized')
    
    tasks = []
    for item in ds['test']:
        task = {
            "task_id": f"mbpp_{item['task_id']}",
            "prompt": item['text'],
            "test_cases": item['test_list'],
            "entry_point": extract_function_name(item['code']),
        }
        tasks.append(task)
    
    with open('./data/mbpp_plus_test.jsonl', 'w') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(tasks)} evaluation tasks")

if __name__ == "__main__":
    prepare_grpo_tasks()
    prepare_eval_tasks()
```

### Step 0.5: 验证模型 + Thinking Mode

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    './models/Qwen3-4B', torch_dtype=torch.float16, device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('./models/Qwen3-4B')

# 测试thinking mode
messages = [
    {'role': 'system', 'content': '你是一个编程助手。'},
    {'role': 'user', 'content': 'Write a function to find the longest common prefix among a list of strings.'}
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True
)
inputs = tokenizer(text, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

# 验证输出包含 <think> tag
assert '<think>' in result, 'ERROR: thinking mode not working'
assert '</think>' in result, 'ERROR: thinking block not closed'

print('=== Full Output ===')
print(result)
print()

# 分离验证
import re
think_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL)
response = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
print(f'Thinking tokens: {len(think_match.group(1).split())}')
print(f'Response tokens: {len(response.split())}')
print('SUCCESS: Thinking mode works, separation clean')
"
```

**检查点**: 输出包含 `<think>` 标签，分离后两部分都有内容

### Step 0.6: 验证DeepSeek API

```bash
python -c "
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY', 'YOUR_KEY'),
    base_url='https://api.deepseek.com'
)

resp = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{'role':'user','content':'Rate this explanation on clarity (1-10): \"I wrote a sort function using quicksort for O(nlogn) average case.\"'}],
    max_tokens=100,
    temperature=0.0
)
print(resp.choices[0].message.content)
print(f'Tokens used: {resp.usage.total_tokens}')
print('SUCCESS: DeepSeek API works')
"
```

**检查点**: API返回正常评分

### Step 0.7: 验证代码执行sandbox

```bash
python -c "
import subprocess
import tempfile
import os

def safe_execute(code: str, test_case: str, timeout: int = 5) -> bool:
    \"\"\"安全执行代码 + 测试用例\"\"\"
    full_code = code + '\n' + test_case
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(f.name)

# 测试
code = 'def add(a, b): return a + b'
test = 'assert add(1, 2) == 3'
assert safe_execute(code, test) == True

code_bad = 'def add(a, b): return a - b'
assert safe_execute(code_bad, test) == False

print('SUCCESS: Code execution sandbox works')
"
```

**检查点**: 正确代码返回True，错误代码返回False

---

## Phase 1: GRPO强化工作层

### Goal

训练模型的coding能力（工作层），同时记录协作层可读性的变化。

### Step 1.1: GRPO训练配置

```yaml
# configs/grpo_config.yaml
model:
  name_or_path: ./models/Qwen3-4B
  torch_dtype: float16
  enable_thinking: true  # 保持thinking mode开启

output_dir: ./checkpoints/cycle_0/model_rl

lora:
  rank: 32
  alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  dropout: 0.05

grpo:
  num_samples_per_prompt: 6
  max_new_tokens: 768
  temperature: 0.7
  top_p: 0.9
  kl_coeff: 0.04
  clip_range: 0.2
  normalize_reward: true

training:
  learning_rate: 2e-5
  num_epochs: 3
  per_device_batch_size: 1
  gradient_accumulation_steps: 6  # effective batch = 4 GPU × 1 × 6 = 24
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  fp16: true
  dataloader_num_workers: 4

data:
  train_file: ./data/mbpp_train.jsonl
  max_tasks: 600  # 从974题中选600题训练，剩余做held-out

reward:
  type: code_execution
  timeout: 10  # seconds per test case
  max_test_cases: 5  # 每题最多跑5个test case (加速)

logging:
  wandb_project: decoupled-collab
  wandb_run: cycle0_grpo
  log_every_n_steps: 10
  save_every_n_steps: 200
```

### Step 1.2: GRPO Reward函数

```python
# src/utils/code_executor.py
"""
Reward函数：只评估代码正确性。
关键设计：reward完全不看协作文本的质量。
这保证了RL只优化"工作层"，协作层的变化是自然的副作用。
"""
import re
import subprocess
import tempfile
import os

def compute_reward(model_output: str, test_cases: list, timeout: int = 10) -> float:
    """
    从模型完整输出中提取代码，执行测试，返回reward。
    
    Args:
        model_output: 模型的完整输出（包含<think>和response）
        test_cases: 测试用例列表，每个是 "assert xxx" 格式
        timeout: 超时时间
    
    Returns:
        0.0 ~ 1.0 的reward分数
    """
    code = extract_code(model_output)
    if not code:
        return 0.0
    
    passed = 0
    for tc in test_cases[:5]:  # 最多5个test case
        if execute_test(code, tc, timeout):
            passed += 1
    
    return passed / min(len(test_cases), 5)


def extract_code(output: str) -> str:
    """从模型输出中提取代码（兼容多种格式）"""
    # 先去掉thinking部分
    output_no_think = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    
    # 尝试提取 ```python ... ```
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', output_no_think, re.DOTALL)
    if code_blocks:
        return '\n'.join(code_blocks)
    
    # 尝试提取 def xxx 开始的代码
    func_match = re.search(r'(def \w+.*?)(?:\n\n|\Z)', output_no_think, re.DOTALL)
    if func_match:
        return func_match.group(1)
    
    return ""


def execute_test(code: str, test_case: str, timeout: int) -> bool:
    """安全地执行代码+测试"""
    full_code = code + '\n\n' + test_case
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)
```

### Step 1.3: 执行GRPO训练

```bash
# 启动训练（预计2-4天）
python src/train_grpo.py --config configs/grpo_config.yaml

# 或用 accelerate 多卡
accelerate launch --num_processes 4 src/train_grpo.py --config configs/grpo_config.yaml
```

**预期时间**: 4×V100-32G, Qwen3-4B, 600 tasks × 6 samples × 3 epochs ≈ **2-3天**

**监控指标**:
- `mean_reward` 应逐步上升（0.3 → 0.5 → 0.7+）
- `kl_divergence` 不应过大（< 15.0）
- GPU利用率应 > 80%

**检查点**:
- `checkpoints/cycle_0/model_rl/adapter_model.safetensors` 存在
- `wandb` 上 reward 曲线上升

### Step 1.4: 验证Phase 1假设（关键！）

```bash
python src/evaluate.py \
    --mode full \
    --base_model ./models/Qwen3-4B \
    --rl_model ./checkpoints/cycle_0/model_rl \
    --eval_data ./data/mbpp_plus_test.jsonl \
    --num_tasks 100 \
    --judge_api deepseek \
    --output ./results/phase1_hypothesis.json
```

评估内容：
1. **Benchmark (pass@1)**: base vs rl 在 MBPP+ 上的代码正确率
2. **可读性**: base vs rl 在协作文本上的可读性得分

**预期结果**:
```json
{
    "benchmark": {
        "base_pass_rate": 0.74,
        "rl_pass_rate": 0.82,
        "delta": "+8%"
    },
    "readability": {
        "base_score": 7.2,
        "rl_score": 5.5,
        "delta": "-1.7"
    },
    "hypothesis_1_verified": true,
    "comment": "RL improves coding but hurts readability"
}
```

**如果假设不成立（RL后可读性没下降）**:
- 方案A: 增加RL epochs / 降低KL约束 → 更aggressive的RL
- 方案B: 检查是否因为4B模型的协作层本来就很简短(没什么可退化的)
- 方案C: 如果确实不退化，这本身是一个有趣的negative result，记录并分析

---

## Phase 2: 收集Traces + 分离Token

### Goal

利用Qwen3的thinking mode天然分离结构，收集Model_RL的输出并提取工作层/协作层。

### Step 2.1: Token分离策略

**核心原则：利用Qwen3 thinking mode的结构化输出，不需要复杂分类算法。**

```
Qwen3-4B (thinking mode) 的输出格式：

<think>                         ← 工作层标记开始
Let me analyze this problem...
The approach should be...
I'll use dynamic programming...
</think>                        ← 工作层标记结束

Here's a function that solves    ← 协作层（面向用户的解释）
this using dynamic programming   
with O(n) time complexity:

```python                        ← 工作层（代码）
def solve(n):
    dp = [0] * (n+1)
    ...
```（结束反引号）               ← 工作层结束

This handles edge cases like     ← 协作层（补充说明）
empty input by returning 0.
```

**分离规则（确定性，无歧义）：**

| 区域 | 分类 | 判定方法 |
|------|------|---------|
| `<think>...</think>` 内 | 工作层 | 正则匹配tag |
| `` ```python...``` `` 内 | 工作层 | 正则匹配代码块 |
| 其余所有文本 | 协作层 | 排除法 |

**对于边界模糊的token（<5%）的处理：一律归入协作层。**
理由：在think tag之外出现的内容，无论是否包含技术细节，都是"面向用户的表达"——这正是我们要优化的对象。

### Step 2.2: 批量收集traces

```bash
python src/collect_traces.py \
    --model ./checkpoints/cycle_0/model_rl \
    --base_model_path ./models/Qwen3-4B \
    --tasks ./data/mbpp_train.jsonl \
    --output ./data/traces/cycle_0_rl_traces.jsonl \
    --num_tasks 2000 \
    --temperature 0.7 \
    --max_new_tokens 768 \
    --enable_thinking true \
    --use_vllm true
```

```python
# src/collect_traces.py 核心逻辑
def collect_single_trace(model, tokenizer, task: dict) -> dict:
    """收集单条trace并分离"""
    messages = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]}
    ]
    
    # 生成（thinking mode开启）
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, 
        add_generation_prompt=True, enable_thinking=True
    )
    output = generate(model, text, max_new_tokens=768, temperature=0.7)
    
    # 分离
    separated = separate_output(output)
    
    # 计算reward
    reward = compute_reward(output, task["test_cases"])
    
    return {
        "task_id": task["task_id"],
        "task_prompt": task["prompt"],
        "full_output": output,
        "thinking": separated["thinking"],       # 工作层-推理
        "code": separated["code"],               # 工作层-代码
        "collaboration": separated["collaboration"],  # 协作层
        "reward": reward,
        "work_trace": separated["thinking"] + "\n[CODE]\n" + separated["code"],
    }


def separate_output(output: str) -> dict:
    """确定性分离：利用Qwen3 thinking mode的tag结构"""
    import re
    
    # 提取thinking
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # 移除thinking
    without_think = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    
    # 提取代码块
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', without_think, re.DOTALL)
    code = '\n'.join(code_blocks).strip()
    
    # 协作层 = 去掉thinking和代码后的剩余文本
    collaboration = re.sub(r'```(?:python)?\s*\n.*?```', '', without_think, flags=re.DOTALL).strip()
    
    return {
        "thinking": thinking,
        "code": code,
        "collaboration": collaboration,
    }
```

**预期时间**: vllm推理，2000条 ≈ **2-4小时**

**检查点**: 
- `data/traces/cycle_0_rl_traces.jsonl` 有 ≥2000 行
- 抽样检查：thinking/code/collaboration三部分都非空

---

## Phase 3: 重生成协作层 + DPO训练

### Goal

用Base Model为RL模型的工作结果重新生成更可读的协作文本，然后用DPO对齐。

### Step 3.1: 用Base Model重新生成协作文本

```bash
python src/regen_collaboration.py \
    --base_model ./models/Qwen3-4B \
    --traces ./data/traces/cycle_0_rl_traces.jsonl \
    --output ./data/dpo_pairs/cycle_0_raw_pairs.jsonl \
    --num_samples 3000 \
    --temperature 0.7 \
    --use_vllm true
```

重生成Prompt：
```python
# src/utils/prompts.py

REGEN_SYSTEM_PROMPT = """你是一个AI编程助手的"表达优化师"。
你的任务是：给定一个编程问题和已完成的解决方案，写一段简洁清晰的说明给用户。"""

REGEN_USER_TEMPLATE = """## 用户的编程需求
{task_prompt}

## 已完成的解决方案（思考过程）
{thinking}

## 已完成的代码
```python
{code}
```

## 请为用户写一段协作消息
要求：
1. 一句话说明你的理解
2. 一句话说明核心思路
3. 如果有重要的设计决策，简要说明
4. 不要重复代码内容
5. 总长度控制在50-150字（中文）或30-80词（英文）"""
```

**关键设计**：
- 用 **Base Model（未RL）** 做重生成，因为它的表达能力未被RL破坏
- 输入的是RL模型的**工作结果**（thinking + code），保证信息准确
- 只需要重写"怎么说"，不改变"做了什么"

### Step 3.2: 用DeepSeek V4 Flash做质量过滤

```bash
python src/filter_pairs.py \
    --raw_pairs ./data/dpo_pairs/cycle_0_raw_pairs.jsonl \
    --output ./data/dpo_pairs/cycle_0_filtered_pairs.jsonl \
    --judge_api deepseek \
    --threshold 6.0 \
    --batch_size 20 \
    --max_concurrent 5
```

```python
# src/utils/api_judge.py

JUDGE_PROMPT = """请评估以下AI编程助手回复的协作质量。

## 用户问题
{task_prompt}

## AI的回复
{collaboration_text}

## 评分标准（每项1-10分）
- clarity: 用户能否快速理解AI做了什么？
- conciseness: 是否简洁不冗余？
- informativeness: 是否传达了关键信息（思路、决策）？
- naturalness: 语言是否自然流畅？

请直接输出JSON（不要其他内容）:
{{"clarity": X, "conciseness": X, "informativeness": X, "naturalness": X, "overall": X}}"""


async def judge_collaboration(client, task_prompt: str, text: str) -> dict:
    """调用DeepSeek V4 Flash评分"""
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                task_prompt=task_prompt,
                collaboration_text=text
            )
        }],
        max_tokens=100,
        temperature=0.0
    )
    return parse_json_score(resp.choices[0].message.content)
```

**过滤逻辑**:
```python
def filter_pair(regen_score: dict, rl_score: dict) -> bool:
    """判断这个pair是否可用于DPO"""
    # 条件1: 重生成的文本得分 ≥ 阈值
    if regen_score["overall"] < 6.0:
        return False
    # 条件2: 重生成的得分 > RL原始文本的得分（chosen确实比rejected好）
    if regen_score["overall"] <= rl_score["overall"]:
        return False
    # 条件3: 差距不能太小（避免噪声pair）
    if regen_score["overall"] - rl_score["overall"] < 0.5:
        return False
    return True
```

**预期**: 3000条raw pairs → 过滤后 ~1500-2000 有效DPO pairs

**检查点**: 过滤后至少1500条有效pairs

### Step 3.3: 构造DPO训练数据

```python
# DPO数据格式
{
    "prompt": "<system>你是编程助手。</system>\n<user>{task_prompt}</user>\n<think>{thinking}</think>\n代码已写好：\n```python\n{code}\n```\n请为用户写一段说明：",
    "chosen": "{regen_collaboration}",    # Base Model重生成的（可读性好）
    "rejected": "{rl_collaboration}"      # RL模型原始的（可读性差）
}
```

**关键**：prompt中包含完整的工作trace（thinking + code）。这样DPO训练的是"给定相同工作结果，如何更好地表达"，而不是"做不同的工作"。

### Step 3.4: DPO训练

```yaml
# configs/dpo_config.yaml
model:
  name_or_path: ./checkpoints/cycle_0/model_rl  # 在RL模型基础上
  torch_dtype: float16

output_dir: ./checkpoints/cycle_0/model_rl_dpo

lora:
  rank: 32
  alpha: 64
  # 复用RL阶段的LoRA adapter，在其基础上继续DPO
  resume_from: ./checkpoints/cycle_0/model_rl

dpo:
  beta: 0.1
  loss_type: sigmoid
  label_smoothing: 0.0
  reference_free: false  # 使用Model_RL作为reference model

training:
  learning_rate: 5e-6    # 比GRPO低，避免破坏工作能力
  num_epochs: 3
  per_device_batch_size: 2
  gradient_accumulation_steps: 4  # effective batch = 4×2×4 = 32
  max_length: 1536
  max_prompt_length: 1024
  warmup_ratio: 0.1
  fp16: true

data:
  train_file: ./data/dpo_pairs/cycle_0_filtered_pairs.jsonl
  eval_split: 0.05  # 5%做验证

logging:
  wandb_project: decoupled-collab
  wandb_run: cycle0_dpo
```

```bash
python src/train_dpo.py --config configs/dpo_config.yaml
```

**预期时间**: 4×V100, 1500 pairs, 3 epochs ≈ **4-8小时**

**监控指标**:
- DPO loss 下降
- `chosen_reward - rejected_reward` margin 增大
- accuracy（chosen reward > rejected reward的比例）上升

**检查点**:
- `checkpoints/cycle_0/model_rl_dpo/adapter_model.safetensors` 存在
- DPO accuracy > 0.7

---

## Phase 4: 评估

### Goal

全面对比三个模型，验证所有假设。

### Step 4.1: 完整评估

```bash
python src/evaluate.py \
    --mode full \
    --models base,rl,final \
    --base_model ./models/Qwen3-4B \
    --rl_model ./checkpoints/cycle_0/model_rl \
    --final_model ./checkpoints/cycle_0/model_rl_dpo \
    --eval_data ./data/mbpp_plus_test.jsonl \
    --lcb_data ./data/lcb_easy.jsonl \
    --num_tasks_benchmark 200 \
    --num_tasks_readability 50 \
    --judge_api deepseek \
    --output ./results/cycle_0_full_eval.json
```

### Step 4.2: 评估指标

**A. Coding能力 (工作层)**:
```python
metrics_benchmark = {
    "mbpp_plus_pass_at_1": float,   # 主指标
    "lcb_easy_pass_at_1": float,    # 辅助（contamination-free）
    "avg_code_length": float,        # 代码简洁度
    "syntax_error_rate": float,      # 语法错误率
}
```

**B. 可读性 (协作层)**:
```python
metrics_readability = {
    "clarity": float,         # 清晰度 (1-10)
    "conciseness": float,     # 简洁度 (1-10)
    "informativeness": float, # 信息量 (1-10)
    "naturalness": float,     # 自然度 (1-10)
    "overall": float,         # 综合分 (1-10)
    
    # 辅助指标
    "avg_collab_length": float,  # 平均协作文本长度(tokens)
    "think_leak_rate": float,    # 推理外泄率（thinking内容出现在response中）
}
```

### Step 4.3: 结果模板

```json
{
    "cycle": 0,
    "timestamp": "2026-08-XX",
    "models": {
        "base": {
            "mbpp_plus_pass1": 0.74,
            "lcb_easy_pass1": 0.65,
            "readability_overall": 7.2,
            "readability_detail": {"clarity": 7.5, "conciseness": 7.0, "informativeness": 7.1, "naturalness": 7.2}
        },
        "rl": {
            "mbpp_plus_pass1": 0.82,
            "lcb_easy_pass1": 0.72,
            "readability_overall": 5.5,
            "readability_detail": {"clarity": 5.8, "conciseness": 4.5, "informativeness": 5.8, "naturalness": 5.9}
        },
        "final": {
            "mbpp_plus_pass1": 0.81,
            "lcb_easy_pass1": 0.71,
            "readability_overall": 7.6,
            "readability_detail": {"clarity": 7.8, "conciseness": 7.5, "informativeness": 7.4, "naturalness": 7.7}
        }
    },
    "hypothesis_results": {
        "H1_rl_improves_coding": {"verified": true, "delta": "+8%"},
        "H2_rl_hurts_readability": {"verified": true, "delta": "-1.7"},
        "H3_dpo_recovers_readability": {"verified": true, "delta": "+2.1"},
        "H4_dpo_preserves_coding": {"verified": true, "delta": "-1%"}
    }
}
```

**检查点**: 4个假设中至少3个verified=true

---

## Phase 5: 迭代循环

### Step 5.1: 启动Cycle 1

```bash
python src/run_pipeline.py \
    --start_model ./checkpoints/cycle_0/model_rl_dpo \
    --cycle_id 1 \
    --config configs/pipeline_config.yaml
```

上一轮的 `model_rl_dpo` 成为新一轮的 base → 再做GRPO → 再做DPO → ...

### Step 5.2: 收敛分析

```
预期趋势：
Cycle 0: coding 74%→82%, readability 7.2→5.5→7.6
Cycle 1: coding 81%→85%, readability 7.6→6.2→7.9
Cycle 2: coding 84%→86%, readability 7.9→6.8→8.1
→ coding逐渐收敛到上限，readability每轮恢复并略有提升
```

如果观察到：
- **coding饱和**: 连续两轮提升<1% → 停止RL，专注DPO
- **readability震荡**: DPO后又被RL打回 → 降低RL的KL约束
- **整体退化**: coding和readability同时下降 → capacity耗尽，停止

---

## 自动化Master脚本

```python
# src/run_pipeline.py
"""
全自动化Pipeline - 支持断点续跑
Usage: python src/run_pipeline.py --config configs/pipeline_config.yaml [--resume]
"""
import json, os, sys
from pathlib import Path
from datetime import datetime
import subprocess

STATE_FILE = "pipeline_state.json"

PHASES = ["phase1_grpo", "phase1_eval", "phase2_collect", 
           "phase3_regen", "phase3_filter", "phase3_dpo", "phase4_eval"]

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {
        "status": "running",
        "current_cycle": 0,
        "current_phase": "phase1_grpo",
        "history": []
    }

def save_state(state):
    state["last_update"] = datetime.now().isoformat()
    json.dump(state, open(STATE_FILE, 'w'), indent=2)

def run_cmd(cmd: str, log_file: str):
    """运行命令，输出写入log文件"""
    print(f"[CMD] {cmd}")
    with open(log_file, 'a') as f:
        result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def get_model_path(cycle: int, phase: str, config: dict) -> str:
    """获取当前应使用的模型路径"""
    if cycle == 0 and phase == "phase1_grpo":
        return config["general"]["base_model"]
    elif phase == "phase1_grpo":
        return f"./checkpoints/cycle_{cycle-1}/model_rl_dpo"
    elif phase in ["phase3_dpo"]:
        return f"./checkpoints/cycle_{cycle}/model_rl"
    else:
        return config["general"]["base_model"]

def main(config_path: str, resume: bool = False):
    config = load_config(config_path)
    state = load_state() if resume else load_state()
    
    for cycle in range(state["current_cycle"], config["general"]["num_cycles"]):
        state["current_cycle"] = cycle
        cycle_dir = f"./checkpoints/cycle_{cycle}"
        os.makedirs(cycle_dir, exist_ok=True)
        log_dir = f"./logs/cycle_{cycle}"
        os.makedirs(log_dir, exist_ok=True)
        
        phases_to_run = PHASES[PHASES.index(state["current_phase"]):]
        
        for phase in phases_to_run:
            state["current_phase"] = phase
            save_state(state)
            
            print(f"\n{'='*60}")
            print(f"  Cycle {cycle} | Phase: {phase}")
            print(f"{'='*60}\n")
            
            if phase == "phase1_grpo":
                model = get_model_path(cycle, phase, config)
                run_cmd(
                    f"accelerate launch --num_processes 4 src/train_grpo.py "
                    f"--config configs/grpo_config.yaml "
                    f"--model {model} --output {cycle_dir}/model_rl",
                    f"{log_dir}/grpo.log"
                )
                
            elif phase == "phase1_eval":
                run_cmd(
                    f"python src/evaluate.py --mode hypothesis_check "
                    f"--base_model {config['general']['base_model']} "
                    f"--rl_model {cycle_dir}/model_rl "
                    f"--output {cycle_dir}/phase1_check.json",
                    f"{log_dir}/phase1_eval.log"
                )
                
            elif phase == "phase2_collect":
                run_cmd(
                    f"python src/collect_traces.py "
                    f"--model {cycle_dir}/model_rl "
                    f"--base_model_path {config['general']['base_model']} "
                    f"--tasks ./data/mbpp_train.jsonl "
                    f"--output ./data/traces/cycle_{cycle}_traces.jsonl "
                    f"--num_tasks 2000 --use_vllm true",
                    f"{log_dir}/collect.log"
                )
                
            elif phase == "phase3_regen":
                run_cmd(
                    f"python src/regen_collaboration.py "
                    f"--base_model {config['general']['base_model']} "
                    f"--traces ./data/traces/cycle_{cycle}_traces.jsonl "
                    f"--output ./data/dpo_pairs/cycle_{cycle}_raw.jsonl "
                    f"--use_vllm true",
                    f"{log_dir}/regen.log"
                )
                
            elif phase == "phase3_filter":
                run_cmd(
                    f"python src/filter_pairs.py "
                    f"--raw_pairs ./data/dpo_pairs/cycle_{cycle}_raw.jsonl "
                    f"--output ./data/dpo_pairs/cycle_{cycle}_filtered.jsonl "
                    f"--judge_api deepseek --threshold 6.0",
                    f"{log_dir}/filter.log"
                )
                
            elif phase == "phase3_dpo":
                run_cmd(
                    f"python src/train_dpo.py --config configs/dpo_config.yaml "
                    f"--model {cycle_dir}/model_rl "
                    f"--dpo_data ./data/dpo_pairs/cycle_{cycle}_filtered.jsonl "
                    f"--output {cycle_dir}/model_rl_dpo",
                    f"{log_dir}/dpo.log"
                )
                
            elif phase == "phase4_eval":
                run_cmd(
                    f"python src/evaluate.py --mode full "
                    f"--models base,rl,final "
                    f"--base_model {config['general']['base_model']} "
                    f"--rl_model {cycle_dir}/model_rl "
                    f"--final_model {cycle_dir}/model_rl_dpo "
                    f"--output ./results/cycle_{cycle}_eval.json",
                    f"{log_dir}/eval.log"
                )
        
        # Cycle完成，重置phase
        state["current_phase"] = "phase1_grpo"
        state["history"].append({
            "cycle": cycle,
            "completed": datetime.now().isoformat(),
            "results": f"./results/cycle_{cycle}_eval.json"
        })
        save_state(state)
    
    state["status"] = "completed"
    save_state(state)
    print("\n✓ All cycles completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.config, args.resume)
```

---

## 错误处理与恢复

| 错误 | 原因 | 恢复方法 |
|------|------|---------|
| CUDA OOM | batch/seq太大 | 减 batch_size，减 max_new_tokens |
| GRPO reward全为0 | 代码执行sandbox问题 | 检查subprocess权限和python path |
| 模型不输出`<think>`tag | thinking mode未正确启用 | 检查 `enable_thinking=True` |
| DPO loss不下降 | lr太高或pairs质量差 | 降lr到1e-6，检查过滤阈值 |
| DeepSeek API 429 | rate limit | 降低并发，加retry+backoff |
| 重生成质量太差 | Base模型能力不足 | 提高temperature做多次采样取最好 |
| RL后可读性没下降 | 4B模型协作文本本来就短 | 增加system prompt要求详细解释 |

### 断点续跑

```bash
# 正常启动
python src/run_pipeline.py --config configs/pipeline_config.yaml

# 中断后恢复
python src/run_pipeline.py --config configs/pipeline_config.yaml --resume
```

---

## 紧急退出条件

出现以下情况，暂停并报告分析：

1. GRPO 3个epoch后 mean_reward < 0.4 → 训练无效
2. DPO后 MBPP+ pass@1 下降 > 5% → DPO破坏了工作能力
3. Phase 1评估中 RL可读性 ≥ Base → 核心假设不成立
4. 连续2个cycle coding提升 < 1% → 已收敛
5. DeepSeek API费用超过 ¥500 → 预算超支

---

## 时间线估算 (4×V100-32G)

| Phase | 任务 | 时间 |
|-------|------|------|
| Phase 0 | 环境+数据+验证 | 0.5天 |
| Phase 1 | GRPO训练 | 2-3天 |
| Phase 1 eval | 假设验证 | 2-4小时 |
| Phase 2 | 收集traces (vllm) | 2-4小时 |
| Phase 3 regen | 重生成 (vllm) | 2-4小时 |
| Phase 3 filter | API过滤 | 1-2小时 |
| Phase 3 DPO | DPO训练 | 4-8小时 |
| Phase 4 | 全面评估 | 4-6小时 |
| **单Cycle总计** | | **~4-5天** |
| **3个Cycle** | | **~2-3周** |

## 预算估算

| 项目 | 数量 | 费用 |
|------|------|------|
| DeepSeek API (过滤) | ~6000 calls × 3 cycles | ~¥180 |
| DeepSeek API (评估) | ~600 calls × 3 cycles | ~¥18 |
| GPU (V100×4) | ~15天 | 自有 |
| **总计** | | **~¥200** |

---

## 版本记录

| 日期 | 变更 | 状态 |
|------|------|------|
| 2026-08-14 | v2: 简化为Qwen3 thinking mode方案，确定MBPP+/LiveCodeBench | NOT_STARTED |
| 2026-08-13 | v2.1: 环境 pin 改为 Qwen3+V100 兼容栈（transformers≥4.51, vllm==0.8.5, torch 2.5/2.6 cu12x, fp16-only） | NOT_STARTED |
