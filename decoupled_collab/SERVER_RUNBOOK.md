# Training server runbook

Deployment prepared on 2026-08-14 for the 4 x V100 server.

## Current deployment state

- The approved server reboot has completed.
- NVIDIA driver `580.173.02` is active.
- PyTorch `2.6.0+cu124` sees four Tesla V100-PCIE-32GB GPUs, all `sm_70`.
- The complete local Qwen3-4B upload passed per-file SHA256 comparison and a
  real GPU generation test.
- DeepSeek authentication was verified with HTTP 200 from the models endpoint;
  the secret remains only in the server `.env` (mode `0600`).
- Real data prepared locally: 596 disjoint MBPP GRPO rows, 378 MBPP+ rows, and
  142 LCB-easy rows. Upload/sync remains pending while the SSH gateway closes
  connections during key exchange.
- CPU-side regression suite after the data-isolation, offline-LCB, launch,
  monitoring, and Trainer checkpoint-resume changes: `59 passed, 1 skipped` on
  2026-08-14. The skipped
  environment test requires the server's full Torch/TRL stack.

## Decision rule: GOAL versus observed state

Values in `GOAL_decoupled_collaboration.md` are planning estimates, not hard
runtime constants. Use measured server behavior and current dataset releases as
the source of truth. Examples:

- Use the actual 596 contamination-free MBPP training rows, not a padded 600.
- Estimate duration after measuring tokens/second and step time in GPU smoke.
- Tune batch size, accumulation, checkpoint cadence, and completion length from
  observed V100 memory headroom and utilization.
- Treat reward/KL/utilization numbers in GOAL as monitoring guidance; stop or
  tune based on actual curves, errors, and evaluation validity.
- Never weaken fail-fast data, thinking-mode, adapter, or test-case checks just
  to match an estimated schedule or target count.

## SSH and tmux connection policy

The pool gateway may rate-limit repeated SSH handshakes. Do not run rapid retry
loops or create a new SSH connection for every status check. Open one SSH
connection and attach to a persistent tmux session:

```bash
ssh -t -p 31520 root@pool.zjuici.com 'tmux new-session -A -s makeai'
```

The local SSH alias can be used if configured:

```bash
ssh -t zjuici 'tmux new-session -A -s makeai'
```

Recommended tmux layout inside that single connection:

```text
window 0: shell / deployment commands
window 1: GRPO log or foreground launcher
window 2: server-local monitoring
```

Useful tmux keys:

- `Ctrl-b c`: create a window
- `Ctrl-b n` / `Ctrl-b p`: next / previous window
- `Ctrl-b ,`: rename the current window
- `Ctrl-b d`: detach without stopping server processes
- `tmux list-sessions`: list sessions after reconnecting

Run monitoring on the server, inside the monitoring window, so it does not
create additional SSH handshakes:

```bash
cd /root/makeAIreadable-20260814/workspace/decoupled_collab
watch -n 60 bash scripts/training_status.sh cycle0_grpo
```

If the gateway closes during key exchange, wait at least 5 minutes before the
next attempt. On repeated failures, increase the interval to 10, then 20
minutes. Use one connection attempt (`ConnectionAttempts=1`) with a finite
timeout; never use a tight automatic reconnect loop. A key-exchange closure
means no shell command reached the server.

## Paths

- Project: `/root/makeAIreadable-20260814/workspace/decoupled_collab`
- Base model: `./models/Qwen3-4B`
- Production Python: `./.venv-prod/bin/python` (Python 3.11)
- Hugging Face cache: `/root/makeAIreadable-20260814/hf_cache`

Run every project command from the project directory because YAML paths are
relative to it.

## Environment

The production environment was created with uv and intentionally does not
contain vLLM. DeepSpeed is also omitted because the current four-process
training path is Accelerate/DDP; installing unused DeepSpeed can require a full
CUDA toolkit during import/build:

Bitsandbytes is prohibited in the FP16 LoRA environment. The tested
`bitsandbytes==0.45.0` imports `triton.ops`, which is absent from the Triton 3.2
bundled with PyTorch 2.6 and caused all four GRPO workers to fail while PEFT
probed optional 8-bit dispatch. No project config requests 4-bit/8-bit loading,
and the setup/launch preflight now fails if bitsandbytes is installed.

```bash
cd /root/makeAIreadable-20260814/workspace/decoupled_collab
.venv/bin/uv venv --python 3.11 .venv-prod
.venv/bin/uv pip install --python .venv-prod/bin/python \
  'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
  'torchaudio==2.6.0+cu124' \
  --index https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
.venv/bin/uv pip install --python .venv-prod/bin/python -r requirements.txt
.venv/bin/uv pip check --python .venv-prod/bin/python
```

Verified package pins include PyTorch 2.6.0+cu124, Transformers 4.52.4,
TRL 0.15.2, PEFT 0.14.0, and Accelerate 1.6.0. Accelerate 1.2.1 is incompatible
with Transformers 4.52.4 because it lacks the `keep_torch_compile` argument on
`Accelerator.unwrap_model`.

TRL 0.15.2 also overrides `_get_train_sampler(self)`, while Transformers 4.52.4
calls it with a dataset argument. `train_grpo.py` contains a narrowly
signature-gated adapter for this exact legacy API and fails on unknown shapes.

## DeepSeek

Copy `.env.example` to `.env`, set mode `600`, and edit the key on the server.
Do not commit or paste the key into logs or chat.

```bash
cp -n .env.example .env
chmod 600 .env
nano .env
set -a
source .env
set +a
```

The server can reach `https://api.deepseek.com`; an unauthenticated probe
returned HTTP 401 as expected. The public API model id is `deepseek-chat`
unless the account uses a gateway with a different id.

## GPU preflight

Before the approved reboot, the loaded NVIDIA kernel module was 535.309.01
while installed user-space libraries were 580.173.02. The reboot resolved that
mismatch. Re-check after any future host maintenance with:

```bash
nvidia-smi
./.venv-prod/bin/python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda_build', torch.version.cuda)
print('available', torch.cuda.is_available(), 'count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

Expected and last verified: four V100 devices with capability `(7, 0)`. Do not
start or resume real training if this changes.

## DPO data construction

After GRPO, do not live-sample traces. `collect_traces.py` harvests
`checkpoints/cycle_N/model_rl/reward_audit/reward_rank*.jsonl`: keep
`reward > 0`, prefer the last third of audit `call` indices, then Base
rewrites collaboration as chosen. `filter_pairs.py` is structural only.
DeepSeek is used in `evaluate.py` (phase1/phase4), not as a train-time
pair gate. Pass `--live` to collect only if you explicitly want the old
second sampling pass.

## Verification and launch

```bash
PYTHON=.venv-prod/bin/python bash scripts/smoke_test.sh
./.venv-prod/bin/python src/prepare_data.py --list-benchmarks
./.venv-prod/bin/python src/prepare_data.py --download
accelerate launch --num_processes 4 src/train_grpo.py \
  --config configs/grpo_config.yaml
```

For a persistent logged launch with stale-PID protection and all preflights:

```bash
bash scripts/launch_grpo_server.sh configs/grpo_config.yaml
bash scripts/training_status.sh cycle0_grpo
```

The 2026-08-14 four-GPU FP16 smoke completed 4/4 steps in 42.6 seconds and
saved a valid 66 MB LoRA adapter plus `checkpoint-4/trainer_state.json`. Peak
observed allocation was about 15.4 GB per V100. Its 64-token completions all
hit the length limit and produced zero code reward, so that smoke proves the
DDP/training/save path but is not a learning-quality test. Use the isolated
768-token reward smoke before a production launch:

```bash
RUN_NAME=gpu_reward_smoke bash scripts/launch_grpo_server.sh \
  configs/grpo_gpu_reward_smoke.yaml
bash scripts/training_status.sh gpu_reward_smoke
```

The first 768-token reward smoke completed in 357.9 seconds (about 89.5 seconds
per step), peaked near 17.4 GB per V100, but every completion hit the limit with
zero reward. A saved single-GPU diagnostic showed an unclosed `<think>` section
with no code. The shared coding prompt now limits reasoning to 300 tokens and
gives runnable fenced code priority. Do not launch production GRPO until the
post-change reward smoke demonstrates closed thinking, extracted code, and at
least some reward variance/non-zero gradients. A follow-up 2048-token diagnostic
also remained inside an unclosed thinking loop, so increasing completion length
alone is rejected. The earlier prompt's explicit request for deep analysis and
alternative exploration was removed; the chat template still receives
`enable_thinking=True`, while the system message now requests one brief approach
and makes a final fenced solution mandatory.

The simplified system message still looped at 768 tokens. Qwen3's official chat
format uses no default system message, so coding now uses one user turn with the
short output contract appended. This keeps the hard thinking switch enabled and
removes the non-standard system-role variable. If this still fails, the next
design is an explicit two-stage thinking budget (force-close the thinking block,
then generate the answer), which requires GRPO log-probability integration and
must not be improvised as a reward-only text rewrite.

The first forced-budget trace closed thinking and emitted executable-looking
code, but scored zero because MBPP training text omitted the required callable
names (`Pair` and `max_chain_length(arr, n)` in the sampled task). MBPP+ eval
prompts already include an assert example. Training/trace prompts now append
only the first training assert as a public interface example; remaining asserts
stay reward-only. This is an actual-data correction, not a reward fallback.

The corrected single-task diagnostic then scored reward `1.0`. GRPO now installs
the same logits processor from config (`thinking_budget_tokens: 256`). The forced
`</think>` control token is excluded through a separate loss mask, so it
contributes no policy loss or KL; `completion_mask` remains intact as the
attention mask. The sampled thinking and all subsequent code tokens remain
normal GRPO actions. Unknown pre-existing logits processors fail fast.

An initial integration incorrectly zeroed TRL's `completion_mask`; TRL also uses
that tensor as the attention mask, so post-close code could not attend to the
forced delimiter. The observed symptom was reward `1.0` but KL `1.49e18` and
NaN gradient norm. That run was stopped after one step. The corrected trainer
keeps the attention mask intact, carries a separate loss mask, zeros inactive
log-ratios before exponentiation, and evaluates KL in FP32.

The corrected four-GPU reward smoke (`gpu_reward_smoke_budget_v2`) completed all
four steps in 300.8 seconds (about 75.2 seconds/step). Rewards by step were
`1.0, 0.0, 0.75, 0.25`; the latter two groups had reward variance and finite,
non-zero gradient norms (`0.183`, `0.191`). Maximum observed KL was `8.26e-6`.
There were no NaNs, OOMs, or leftover GPU processes, and both the final LoRA and
`checkpoint-4` contain valid adapter files. At that measured rate, the reference
596-task, three-epoch setup is roughly a 37-hour run, so treat GOAL counts as a
budget ceiling and scale only after reviewing completion audits and a longer
pilot.

Every real GRPO run writes per-rank completion audits under
`OUTPUT_DIR/reward_audit/reward_rankN.jsonl`. Each row records task ID, exact
completion, code reward, and basic think/fence diagnostics without invoking the
judge API or rerunning the code tests. Use these files to distinguish zero reward
caused by bad code from truncation, missing fences, or an unclosed think block.
Generation also stops each sequence as soon as the second Markdown fence after
`</think>` closes the required Python block (`stop_after_code_fence: true`). This
prevents already-correct solutions from wasting the remaining token budget on
post-code explanation. The controller ignores fences emitted inside the thinking
section. DDP unused-parameter detection is disabled because the validated LoRA
forward uses every trainable parameter and PyTorch reported that the extra graph
traversal was unnecessary.

The first fence-stop implementation assumed triple backticks were always one
token. Qwen can instead emit a closing fence as split tokens (for example
`"``" + "`\n"`), so that implementation did not actually stop. The corrected
controller scans decoded token pieces, recognizes fences across token boundaries,
and explicitly masks all padding after the closing fence. Its one-task A/B kept
reward `0.75`, reduced mean effective completion length from `768` to `621`, and
ran in 78.2 seconds versus the 89.6-second baseline. A 640-token cap was rejected:
it truncated a correct sample that closed around token 642, reduced reward to
`0.50`, and did not improve wall time reliably. Keep 768 as the production
ceiling until the separate 704-token four-task comparison has passed.

Evaluation must use the same generation contract as GRPO. `evaluate.py` therefore
forces `</think>` at `--thinking_budget_tokens` (default 256), stops after the
closed Python fence, and adds the first assert as a public interface example.
Running evaluation without this controller can leave Qwen3-4B in an unclosed
thinking loop and produce a falsely low pass@1.

After a confirmed interrupted GRPO run, resume the highest valid
`checkpoint-N` (a directory must contain `trainer_state.json`):

```bash
RESUME=1 bash scripts/launch_grpo_server.sh configs/grpo_config.yaml
```

The resume request is fail-fast: if no valid Trainer checkpoint exists it
refuses to silently restart from step zero. Pipeline `--resume` passes the same
checkpoint-resume request only to the first unfinished GRPO/DPO phase.

The server's current network path resolves or routes Hugging Face incorrectly:
`huggingface.co` and `hf-mirror.com` time out, while PyPI and DeepSeek work.
Prepare the three benchmark JSONL files on a machine with working Hugging Face
access and upload them, or configure a reliable server proxy before the real
pipeline. The real pipeline correctly refuses synthetic or missing benchmark
files.

The generated GRPO file must also be disjoint from MBPP+. Preparation merges
the 974 MBPP-full rows and excludes the 378 EvalPlus MBPP+ IDs, producing 596
training rows with the current releases. `run_pipeline.py` fails before training
if normalized IDs such as `mbpp_602` and `Mbpp/602` overlap.

LiveCodeBench's default `release_latest` builder downloads every historical
JSONL shard and can consume several GB. Prefer a pinned/local source. If a raw
LCB shard is already cached locally, convert it without any further network use:

```bash
python src/prepare_data.py --skip-train --skip-mbpp-plus \
  --lcb-source-jsonl /path/to/test.jsonl
```

Only easy rows, prompts, metadata needed for call tests, and public tests are
written to `data/lcb_easy.jsonl`; private tests are intentionally discarded.

## Resume and logs

```bash
./.venv-prod/bin/python src/run_pipeline.py \
  --config configs/pipeline_config.yaml
./.venv-prod/bin/python src/run_pipeline.py \
  --config configs/pipeline_config.yaml --resume
```

- State: `pipeline_state.json` (`current_phase` means the next phase to run)
- Logs: `logs/cycle_N/`
- Checkpoints: `checkpoints/cycle_N/`
- Results: `results/`

Never resume a real run from a directory containing `DRY_RUN_PLACEHOLDER`.
