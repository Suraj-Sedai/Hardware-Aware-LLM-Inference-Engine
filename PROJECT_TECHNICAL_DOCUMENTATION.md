# Hardware-Aware LLM Inference Engine: Technical Documentation

## 1. Project Intent

This repository is a compact LLM inference engine prototype focused on inference-time systems ideas rather than model training. The codebase is organized around a simplified GPT-style model and several inference-engine concerns:

- model forward execution
- autoregressive generation
- KV cache management
- sampling
- scheduling primitives
- profiling and benchmark measurement
- optimization experiments such as quantization and flash attention

The implementation is deliberately small. A number of modules are fully functional for experimentation, while some are scaffolds for future work.

## 2. High-Level Architecture

At a high level, the runtime path is:

1. Input text is optionally tokenized by `SimpleTokenizer`.
2. Token IDs are fed into `GPT`.
3. `GPT` runs token embedding, positional embedding, and a stack of `TransformerBlock`s.
4. Each `TransformerBlock` runs `SelfAttention` and `MLP`.
5. `SelfAttention` optionally writes keys and values into a KV cache through `KVCacheManager.get_for_attention(...)`.
6. `InferenceController.generate(...)` performs:
   - prefill over the prompt
   - iterative decode over new tokens
   - sampling through `sample_top_k` or greedy argmax
   - latency and memory measurement
7. Profiling helpers convert raw generation outputs into benchmark result schemas.
8. Benchmark scripts run experiment sweeps and visualization scripts render plots into `results/`.

## 3. End-to-End Pipeline

### 3.1 Text to Tokens

If the demo tokenizer is used:

- `SimpleTokenizer.build_vocab(corpus)` creates a small word-level vocabulary.
- `SimpleTokenizer.encode(text)` converts whitespace-separated words to token IDs.
- unknown tokens map to `<UNK>`.

This is only for demos. The engine logic is independent from the tokenizer choice as long as `input_ids` are integer token tensors.

### 3.2 Prefill Phase

The prefill phase processes the full prompt sequence.

- `InferenceController.generate(...)` allocates `output_tokens` of shape `(B, prompt_len + max_new_tokens)`.
- It calls `model(input_ids, active_kv_cache, latency_tracker=...)`.
- Inside `GPT.forward(...)`, positions are computed using `kv_cache.curr_len` so decode can continue from cached sequence length.
- Each transformer layer:
  - layer-norms the input
  - runs self-attention
  - updates or reads KV cache
  - runs MLP
- After all layers, `GPT` increments `kv_cache.curr_len` by the processed token count.
- The controller records this prefill latency as TTFT-like first-step latency.

### 3.3 Decode Phase

The decode phase iteratively generates new tokens.

- If KV cache is enabled, only the latest token slice `output_tokens[:, curr_len - 1:curr_len]` is forwarded each step.
- If KV cache is disabled, the full prefix `output_tokens[:, :curr_len]` is recomputed each step.
- The controller samples the next token from `logits[:, -1, :]`.
- The token is written into the preallocated `output_tokens` tensor.
- The loop stops early if all sampled tokens are `0`, treated here as EOS by convention.

### 3.4 Metrics and Benchmark Output

The generation call returns:

- generated tokens
- per-step latencies
- phase timings
- peak GPU memory

Profiling helpers then derive:

- TTFT
- average TPOT
- TPOT percentiles
- throughput
- phase breakdowns
- standardized result records for experiment scripts

## 4. Folder Breakdown

### 4.1 `src/`

Primary implementation code lives here.

Subpackages:

- `tokenizer/`: simple demonstration tokenizer
- `model_core/`: GPT, transformer block, attention, MLP
- `kv_cache/`: contiguous cache implementation plus placeholders for paged and quantized caches
- `inference/`: generation controller and workload simulator
- `sampling/`: token selection methods
- `scheduler/`: queueing and batching primitives
- `profiling/`: device, latency, memory, and benchmark result helpers
- `optimizations/`: quantization, flash attention, fused ops

### 4.2 `configs/`

Small configuration helpers for model sizes.

### 4.3 `benchmarks/`

Experiment runners and plotting utilities for performance studies.

### 4.4 `tests/`

Unit tests validating the migration to `InferenceController` and metric behavior.

### 4.5 `notebooks/`

An exploratory notebook that duplicates much of the engine logic inline for experimentation.

### 4.6 `results/`

Generated benchmark plot outputs. These are artifacts, not source modules.

### 4.7 `.dist/`

Build/output-style directory. Not used by the engine code directly.

### 4.8 `.git/`

Git metadata. Not part of the application runtime.

## 5. File-by-File Documentation

### 5.1 Repository Root Files

#### `README.md`

Purpose:

- project summary
- high-level architecture list
- rough folder map

Notes:

- it describes the intended engine shape accurately
- usage instructions are still minimal

#### `.gitignore`

Purpose:

- excludes virtualenvs, Python artifacts, notebooks checkpoints, IDE files, logs, weights, datasets, profiling outputs, secrets, and generated directories

Important detail:

- it ignores `*.json`, which means benchmark JSON results would be ignored by default
- it does not ignore the existing `results/*.png`, so plots can remain versioned if desired

#### `requirements.txt`

Purpose:

- dependency list for PyTorch, Transformers ecosystem, Triton, Flash Attention, bitsandbytes, profiling, visualization, and test tooling

Observation:

- dependencies reflect an ambition larger than the currently implemented code
- some packages are not yet directly exercised in the source tree

#### `pyproject.toml`

Purpose:

- basic setuptools build definition
- package metadata such as project name and Python version

#### `setup.py`

Purpose:

- setuptools packaging entry

Behavior:

- calls `find_packages()`
- prints discovered packages before invoking `setup(...)`

Observation:

- the debug print is useful during packaging setup but usually removed in polished packages

#### `debug_kv_cache.py`

Purpose:

- top-level standalone profiling script to compare cached vs non-cached decode behavior on a larger GPT instance

What it does:

- creates a large model
- times batch forward passes
- times cached single-token decode
- compares end-to-end cached and no-cache generation
- inspects cache state and non-zero values

Note:

- this file is duplicated almost exactly in `src/kv_cache/debug_kv_cache.py`

#### `PROJECT_TECHNICAL_DOCUMENTATION.md`

Purpose:

- this document

### 5.2 `configs/`

#### `configs/__init__.py`

Purpose:

- marks `configs` as a package

#### `configs/model_config.py`

Purpose:

- stores tiny GPT configuration presets

Key items:

- `DEFAULT_CONFIG`
- `get_config(size="small")`

Behavior:

- returns `small`, `medium`, or `large` dictionaries
- uses shallow dictionary expansion to adjust `dim`, `n_heads`, and `n_layers`

Role in project:

- convenience helper for experiments and demos

### 5.3 `src/`

#### `src/__init__.py`

Purpose:

- exposes a simplified package API:
  - `GPT`
  - `KVCacheManager`
  - `SimpleTokenizer`
  - `InferenceController`
  - `sample_top_k`

Role:

- lets consumers import core pieces from `src`

### 5.4 `src/tokenizer/`

#### `src/tokenizer/__init__.py`

Purpose:

- re-exports `SimpleTokenizer`

#### `src/tokenizer/simple_tokenizer.py`

Main class: `SimpleTokenizer`

Responsibility:

- basic word-level tokenization for demos and notebook experiments

Methods:

- `__init__(vocab=None)`: initializes vocabulary with `<PAD>` and `<UNK>` if none is provided
- `build_vocab(corpus, max_vocab=100)`: counts whitespace-separated words and builds a capped vocab
- `encode(text)`: converts words to token IDs
- `decode(token_ids)`: converts token IDs back to words
- `__len__()`: returns vocab size

Important characteristics:

- whitespace tokenization only
- no subword logic
- no BOS/EOS handling other than whatever token IDs the caller uses
- suitable for toy inference only

### 5.5 `src/model_core/`

This package contains the neural network itself.

#### `src/model_core/__init__.py`

Purpose:

- re-exports `SelfAttention`, `MLP`, `TransformerBlock`, and `GPT`

#### `src/model_core/attention.py`

Main class: `SelfAttention`

Responsibility:

- multi-head self-attention with optional KV cache integration

Initialization:

- stores `n_heads` and `head_dim`
- creates four projections:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `out_proj`

`forward(x, kv_cache=None, layer_id=None)`:

1. projects input into Q, K, V
2. reshapes to `(B, n_heads, T, head_dim)`
3. if cache is present, calls `kv_cache.get_for_attention(layer_id, k, v)`
4. computes scaled dot-product attention
5. applies softmax
6. multiplies by values
7. merges heads back to `(B, T, D)`
8. applies `out_proj`

Returns:

- output tensor
- `k`
- `v`

Design note:

- the returned `k` and `v` are no longer used by `GPT`, because cache update happens inside `get_for_attention(...)`

Limitation:

- no causal mask is applied explicitly
- because decode is incremental and cache order is prefix-only, behavior is still reasonable for autoregressive experiments, but full-prompt training-style masking is absent

#### `src/model_core/mlp.py`

Main class: `MLP`

Responsibility:

- feed-forward network within each transformer block

Structure:

- `fc1`: input to hidden
- `GELU`
- `fc2`: hidden back to model dimension

Behavior:

- standard transformer MLP with hidden size defaulting to `4 * dim`

#### `src/model_core/transformer.py`

Main class: `TransformerBlock`

Responsibility:

- one residual transformer block

Structure:

- `ln1`
- `SelfAttention`
- `ln2`
- `MLP`

`forward(x, kv_cache=None, layer_id=None)`:

1. layer-normalize input
2. run attention
3. add residual
4. layer-normalize again
5. run MLP
6. add residual

Returns:

- updated hidden state
- attention-produced `K_new`
- attention-produced `V_new`

Observation:

- `K_new` and `V_new` are part of an older explicit-cache-update design and are currently ignored by higher layers

#### `src/model_core/gpt.py`

Main class: `GPT`

Responsibility:

- top-level language model for inference experiments

Architecture:

- token embedding
- positional embedding
- repeated `TransformerBlock`s
- final layer norm
- output projection to vocabulary logits

`forward(input_ids, kv_cache=None, latency_tracker=None)`:

1. reads batch and sequence dimensions
2. computes starting position from `kv_cache.curr_len`
3. builds position IDs for current token slice
4. sums token and positional embeddings
5. iterates over transformer blocks
6. optionally marks per-layer latency boundaries
7. increments `kv_cache.curr_len` after all layers finish
8. returns logits over the vocabulary

Important design choice:

- cache length increments once per model forward, not once per layer

Implication:

- `KVCacheManager.get_for_attention(...)` writes layer-local K/V using the same current sequence offset across all layers, then `GPT` advances the global length only after the full forward pass

Limitation:

- no explicit `torch.no_grad()` inside the module; callers are expected to use eval/no-grad contexts during inference benchmarks

### 5.6 `src/kv_cache/`

#### `src/kv_cache/__init__.py`

Purpose:

- re-exports cache implementations

Exports:

- `KVCacheManager`
- `PagedKVCache`
- `QuantizedKVCache`

#### `src/kv_cache/contiguous_cache.py`

Main class: `KVCacheManager`

Responsibility:

- contiguous, preallocated KV cache for all layers

Internal state:

- `self.K`: list of key tensors, one per layer
- `self.V`: list of value tensors, one per layer
- `self.curr_len`: current cached sequence length shared across layers

Tensor shape:

- each layer cache tensor is `(batch_size, n_heads, max_seq_len, dim_head)`

Methods:

- `__init__(...)`: allocates full cache upfront
- `update(layer_id, K_new, V_new)`: writes new keys/values into the active slice
- `get_for_attention(layer_id, K_new, V_new)`: writes new data and returns a prefix view including old and new tokens
- `get(layer_id)`: returns current cached prefix for a layer
- `reset()`: sets `curr_len = 0`

Why it matters:

- avoids repeated `torch.cat(...)`
- makes decode cheap by reusing prior prefix K/V
- returns views instead of copied tensors

Important implementation detail:

- `reset()` does not zero the allocated buffers
- it only resets logical length
- this is fine because subsequent writes overwrite the active region

#### `src/kv_cache/paged_cache.py`

Main class: `PagedKVCache`

Status:

- placeholder only

Declared methods:

- `__init__`
- `update`
- `get_for_attention`
- `reset`

Intended role:

- reduce fragmentation and support page/block-based cache management similar to real serving systems

Current reality:

- no implementation yet

#### `src/kv_cache/quantized_cache.py`

Main class: `QuantizedKVCache`

Status:

- placeholder only

Declared methods:

- `__init__`
- `update`
- `get_for_attention`
- `reset`

Intended role:

- reduce KV memory footprint via 8-bit or 4-bit representations

Current reality:

- no implementation yet

#### `src/kv_cache/debug_kv_cache.py`

Purpose:

- same standalone debug/profiling script as repository root `debug_kv_cache.py`

Observation:

- this duplication suggests the project evolved quickly during experimentation
- one copy could be removed later to avoid drift

### 5.7 `src/inference/`

#### `src/inference/__init__.py`

Purpose:

- re-exports generation helpers and simulator pieces

#### `src/inference/controller.py`

Main class: `InferenceController`

This is the most important orchestration module in the repo.

Responsibility:

- manage generation loop
- coordinate model and KV cache
- collect latency and memory metrics
- support benchmark experiments through a single generation path

Initialization:

- stores `model`
- stores `kv_cache`
- stores `device`
- stores optional `ablation_flags`
- creates a `LatencyTracker`

Key method: `generate(...)`

Arguments:

- `input_ids`
- `max_new_tokens`
- `temperature`
- `top_k`
- `sample_fn`
- `use_kv_cache`

Flow:

1. derive prompt length and total output length
2. choose active cache or disable it
3. reset latency-tracker state
4. preallocate `output_tokens`
5. choose sampling function:
   - greedy if `top_k == 1`
   - top-k otherwise
6. reset memory stats
7. run prefill
8. loop decode token by token
9. sample next token
10. write token into output buffer
11. stop on EOS token `0` if all batch items hit it
12. synchronize latency tracker
13. return structured generation output

Why this module matters:

- it replaced older generation code that repeatedly grew tensors with `torch.cat(...)`
- it gives the repo one main inference execution path

Return schema:

- `"tokens"`
- `"latencies"`
- `"phase_times"`
- `"peak_memory_mb"`

Secondary method: `warmup(...)`

Purpose:

- run a few warmup trials before real benchmarking

Behavior:

- resets cache between warmup runs

Limitations:

- no batching scheduler integration yet
- no streaming callback
- EOS handling assumes token `0`
- `ablation_flags` is currently stored but not used
- current memory after generation is not returned even though metrics schema supports it

#### `src/inference/generate.py`

Functions:

- `generate(...)`
- `generate_greedy(...)`

Purpose:

- thin compatibility wrappers around `InferenceController`

Role:

- preserves a simple function-style API while centralizing logic in the controller

Note:

- docstrings still say temperature/top-k are "currently unused", but the controller now uses them to choose sampling logic

#### `src/inference/workload_simulator.py`

Main class: `WorkloadSimulator`

Responsibility:

- generate synthetic request streams with arrival times and varying prompt/decode lengths

Methods:

- `__init__(avg_arrival_rate=1.0, prompt_range=(10, 50), decode_range=(20, 100))`
- `generate_requests(num_requests)`

Request format:

- `id`
- `arrival_time`
- `prompt_len`
- `decode_len`

Function: `run_workload(controller, requests, tokenizer=None)`

Responsibility:

- runs a list of synthetic requests through a controller sequentially

Behavior:

- creates random prompt token tensors
- resets cache per request
- generates outputs
- attaches request metadata

Limitation:

- arrivals are generated, but not actually simulated with concurrency or overlap
- this is a sequential workload runner, not a real event-driven serving simulator

### 5.8 `src/sampling/`

#### `src/sampling/__init__.py`

Purpose:

- re-exports sampling functions

#### `src/sampling/strategies.py`

Functions:

- `sample_top_k(logits, k=50, temperature=1.0)`
- `sample_greedy(logits)`
- `sample_top_p(logits, p=0.9, temperature=1.0)`

Responsibilities:

- select next token from model logits

Implementation notes:

- `sample_top_k` clips `k` to vocab size, softmaxes top-k logits, and samples among them
- `sample_greedy` does argmax
- `sample_top_p` sorts logits, builds a cumulative probability mask, suppresses tails with `-inf`, then samples

Role in project:

- decouples token selection policy from the controller

### 5.9 `src/scheduler/`

This package sketches serving-system building blocks but is not yet integrated into inference.

#### `src/scheduler/__init__.py`

Purpose:

- package marker
- contains TODO note for future scheduler implementation

#### `src/scheduler/request_queue.py`

Main class: `RequestQueue`

Responsibility:

- simple FIFO queue for incoming requests

Methods:

- `add_request(request)`
- `pop_request()`
- `size()`

#### `src/scheduler/batching.py`

Main class: `ContinuousBatching`

Responsibility:

- track active requests and admit new ones up to `max_batch_size`

Methods:

- `can_add()`
- `add_request(request)`
- `remove_finished()`

Expectation:

- requests inserted here are expected to implement `is_finished()`

Current status:

- not connected to the inference controller
- useful as a conceptual placeholder for a future serving loop

#### `src/scheduler/chunked_prefill.py`

Main class: `ChunkedPrefill`

Responsibility:

- split long prompts into chunks

Method:

- `get_chunks(input_ids)`

Role:

- models the prefill chunking idea used in production inference engines

Current status:

- utility exists, but nothing in the controller calls it yet

### 5.10 `src/profiling/`

#### `src/profiling/__init__.py`

Purpose:

- re-exports device, memory, latency, metric, and benchmark helpers

#### `src/profiling/memory_tracker.py`

Functions:

- `get_gpu_memory_mb()`
- `get_gpu_max_memory_mb()`
- `reset_gpu_memory_stats()`
- `get_memory_stats()`

Responsibility:

- wrap PyTorch CUDA memory APIs with CPU-safe fallbacks

Behavior:

- returns zeros or `{}` when CUDA is unavailable

#### `src/profiling/latency_tracker.py`

Main class: `LatencyTracker`

Responsibility:

- record phase and layer durations

Design:

- on CUDA, uses `torch.cuda.Event`
- on CPU, uses `time.perf_counter()`

Methods:

- `start_phase(phase_name)`
- `end_phase(phase_name)`
- `start_layer(layer_id)`
- `end_layer(layer_id)`
- `synchronize()`
- `get_times()`

Important detail:

- per-layer timings are stored in `phase_times` using names like `layer_0`, `layer_1`
- `layer_times` exists as a field but is not currently used

#### `src/profiling/metrics.py`

Functions:

- `calculate_metrics(...)`
- `normalize_phase_times_ms(...)`
- `build_benchmark_result(...)`
- `format_benchmark_result(...)`

This file defines the benchmark result schema used throughout the repo.

`calculate_metrics(...)` computes:

- TTFT from first latency entry
- TPOT average
- TPOT p50/p95/p99
- throughput

`build_benchmark_result(...)` adds:

- experiment name
- model name
- device
- timestamp
- config
- normalized metric fields
- phase timing dictionary
- prompt/generated/output token counts
- `extras`

Why this matters:

- it is the contract between raw generation runs and experiment reporting

#### `src/profiling/benchmark.py`

Functions:

- `get_device()`
- `get_gpu_utilization()`
- `get_gpu_memory_mb()`
- `get_cpu_utilization()`
- `benchmark_generation(...)`

Responsibilities:

- pick execution device
- collect host/GPU utilization
- run warmup and benchmark flow through `InferenceController`
- produce standardized benchmark results

Notable behavior:

- `get_device()` tests CUDA allocation before trusting CUDA availability
- `benchmark_generation(...)` imports `InferenceController` lazily inside the function

### 5.11 `src/optimizations/`

This package contains optimization experiments and lower-level alternatives to the baseline model path.

#### `src/optimizations/__init__.py`

Purpose:

- re-exports flash attention, quantization, and fused attention helpers

#### `src/optimizations/fused_kernels.py`

Functions:

- `fused_attention(q, k, v, mask=None)`
- `fused_gelu(x)`
- `fused_layer_norm(x, weight, bias, eps=1e-5)`

Class:

- `FusedMLP`

Purpose:

- demonstrate fused operation variants implemented in PyTorch

Notes:

- this is not a custom CUDA or Triton kernel yet
- it is more of a fused-op approximation at the Python/PyTorch level

`FusedMLP`:

- same role as `MLP`
- uses `fused_gelu` between two linear layers

#### `src/optimizations/flash_attention.py`

Function:

- `flash_attention_available()`

Main class:

- `FlashAttention`

Responsibility:

- provide an attention module that uses `flash_attn_func` when available on CUDA and otherwise falls back to standard attention

Flow:

1. project Q/K/V
2. reshape into flash-attention-friendly layout `(B, T, n_heads, head_dim)`
3. if cache exists, transpose to cache layout, write/read cache, then transpose back
4. use flash attention if available and CUDA tensor
5. else run `_standard_attention(...)`
6. project output

Important caveat:

- this module is not wired into `GPT` or `TransformerBlock`
- it exists as an alternative implementation, not an active one

Subtle difference vs baseline:

- this class increments `kv_cache.curr_len` inside attention when `layer_id` is the last layer
- baseline `GPT` increments cache length after the block loop
- because it is not currently integrated, this inconsistency does not break active code, but it should be unified before adoption

#### `src/optimizations/quantization.py`

Functions:

- `get_model_size_mb(model)`
- `quantize_model_dynamic(model)`
- `quantize_model_static(model, calibration_data=None)`
- `quantize_weights_int8(weight)`
- `dequantize_weights_int8(quantized, scale)`
- `quantize_model(model, use_pytorch_quantization=True)`
- `_quantize_model_recursive(model)`
- `dequantize_model(model)`

Class:

- `QuantizedLinear`

Responsibility:

- implement multiple quantization paths for inference experiments

Quantization modes:

- PyTorch dynamic quantization on CPU
- PyTorch static quantization with optional calibration
- manual INT8 quantization using `QuantizedLinear`

`QuantizedLinear` behavior:

- stores `int8` weights plus scale
- caches a dequantized float copy on first use per device
- uses `nn.functional.linear(...)` for forward pass

Why this matters:

- manual mode reduces model memory but may not improve speed
- CPU dynamic quantization can use optimized kernels

Current project usage:

- experiment 4 uses the manual INT8 path when on CUDA
- dynamic quantization support exists but is not currently benchmarked in the provided experiment loop

### 5.12 `benchmarks/`

This folder turns the engine into a measurement framework.

#### `benchmarks/__init__.py`

Purpose:

- re-exports `run_all_experiments`, `plot_results`, and `save_plots`

#### `benchmarks/run_benchmarks.py`

Purpose:

- small smoke benchmark runner

Behavior:

- selects device
- constructs a tiny GPT
- loops over sequence lengths
- runs `benchmark_generation(...)`
- prints throughput and latency
- prints full JSON result

#### `benchmarks/benchmark_prefill.py`

Function:

- `run_prefill_benchmark(model_config, batch_size, prompt_len, device="cuda")`

Purpose:

- measure a prompt-heavy inference run through the shared benchmark path

Note:

- although named "prefill", it still calls `benchmark_generation(..., max_new_tokens=1)`, so it includes one decode step as part of the run

#### `benchmarks/benchmark_decode.py`

Function:

- `run_decode_benchmark(model_config, batch_size, decode_len, device="cuda")`

Purpose:

- measure decode behavior by starting from a single-token prompt and generating `decode_len` tokens

#### `benchmarks/benchmark_workloads.py`

Function:

- `run_workload_benchmark(model_config, num_requests, device="cuda")`

Purpose:

- simulate a stream of requests and aggregate metrics over all of them

Flow:

1. create model and cache
2. build `InferenceController`
3. generate synthetic requests with `WorkloadSimulator`
4. run them sequentially via `run_workload(...)`
5. aggregate TTFT/TPOT and totals
6. convert to standardized result schema

Important note:

- this is a request-sequence benchmark, not a concurrent serving benchmark

#### `benchmarks/exp1_sequence_length.py`

Function:

- `run_sequence_length_experiment(...)`

Purpose:

- sweep sequence lengths and measure performance trends

Behavior:

- creates a fresh model/cache per sequence length
- warms up
- runs generation
- builds result records
- optionally saves JSON to `results/exp1_sequence_length.json`

#### `benchmarks/exp2_batch_size.py`

Function:

- `run_batch_size_experiment(...)`

Purpose:

- sweep batch sizes with fixed sequence length

Behavior:

- otherwise follows the same controller-driven benchmark path as experiment 1

#### `benchmarks/exp3_kv_cache.py`

Functions:

- `get_cache_memory_mb(kv_cache)`
- `run_kv_cache_experiment(...)`

Purpose:

- compare cached generation versus no-cache generation
- quantify cache memory usage

Behavior:

- for each sequence length:
  - benchmark with KV cache on
  - benchmark with KV cache off
  - compute speedup
  - save both result variants

This is the clearest systems experiment in the repo because it directly tests the central optimization idea.

#### `benchmarks/exp4_quantization.py`

Function:

- `run_quantization_experiment(...)`

Purpose:

- compare FP32 and manual INT8 quantized models across several model sizes

Behavior:

- builds several GPT sizes
- measures FP32 baseline
- if on CUDA, clones weights into a manual-INT8 model and benchmarks that too
- saves standardized results

Observation:

- despite the file docstring mentioning CPU dynamic quantization, the main loop does not currently benchmark that path

#### `benchmarks/run_all.py`

Function:

- `run_all_experiments(output_dir="results")`

Purpose:

- master entrypoint to run experiments 1 through 4
- write aggregate JSON
- render plots

Output behavior:

- creates `results/run_<timestamp>/`
- saves `results.json`
- generates plot PNGs

#### `benchmarks/visualize.py`

Purpose:

- plotting and summary visualization utilities

Main functions:

- `plot_sequence_length_results(...)`
- `plot_batch_size_results(...)`
- `plot_kv_cache_results(...)`
- `plot_quantization_results(...)`
- `plot_all_results(...)`
- `plot_summary(...)`
- aliases `save_plots(...)` and `plot_results(...)`

Behavior:

- uses matplotlib if installed
- safely skips plotting if matplotlib is unavailable
- reads the standardized result schema instead of experiment-specific ad hoc formats

### 5.13 `tests/`

#### `tests/test_migration.py`

Main class: `TestInferenceMigration`

Purpose:

- verify behavior after moving generation logic into `InferenceController`

Test cases:

- `test_controller_generation_shape`
- `test_metrics_consistency`
- `test_cache_reset`
- `test_peak_memory_reporting`

What these tests validate:

- output token shape is stable
- metric calculations are numerically correct
- cache reset resets logical length
- peak memory field is present

Coverage note:

- tests are focused on controller migration and profiling outputs
- they do not yet cover scheduler, flash attention, quantization module internals, or workload simulation

### 5.14 `notebooks/`

#### `notebooks/00_overall_experiments.ipynb`

Purpose:

- exploratory notebook version of the project

Observed content:

- device helper logic
- inline `KVCacheManager`
- inline `SelfAttention`, `MLP`, `TransformerBlock`, and `GPT`
- sampling
- generation routine
- minimal test case
- tokenizer demo

Interpretation:

- this notebook appears to be an earlier or parallel prototyping environment from which parts of `src/` were later extracted

### 5.15 `results/`

#### `results/run_20260330_111107/`
#### `results/run_20260330_114018/`
#### `results/run_20260330_120144/`
#### `results/run_20260330_121041/`
#### `results/run_20260403_141737/`
#### `results/run_20260404_094713/`

Each run directory contains:

- `exp1_sequence_length.png`
- `exp2_batch_size.png`
- `exp3_kv_cache.png`
- `exp4_quantization.png`
- `summary.png`

Purpose:

- saved visualization artifacts from benchmark runs

These are generated outputs, not source modules.

## 6. Module Relationships

### Runtime dependency graph

Core path:

- `src/tokenizer/simple_tokenizer.py` -> produces demo token IDs
- `src/inference/controller.py` -> drives generation
- `src/model_core/gpt.py` -> model forward
- `src/model_core/transformer.py` -> block composition
- `src/model_core/attention.py` -> cache-aware attention
- `src/model_core/mlp.py` -> feed-forward network
- `src/kv_cache/contiguous_cache.py` -> cache state and prefix views
- `src/sampling/strategies.py` -> next-token selection
- `src/profiling/latency_tracker.py` and `src/profiling/memory_tracker.py` -> timing and memory
- `src/profiling/metrics.py` -> standardized benchmark records

Experiment path:

- benchmark scripts create model + cache + controller
- benchmarking helpers call controller
- metrics helpers format results
- visualization utilities plot outputs

Future-serving path sketched but not integrated:

- `src/scheduler/request_queue.py`
- `src/scheduler/batching.py`
- `src/scheduler/chunked_prefill.py`

Alternative optimization path not integrated:

- `src/optimizations/flash_attention.py`
- `src/optimizations/fused_kernels.py`
- `src/optimizations/quantization.py`

## 7. Key Classes and Their Roles

### Production-path classes

- `SimpleTokenizer`: toy tokenization for demos
- `SelfAttention`: computes attention and interacts with KV cache
- `MLP`: feed-forward sublayer
- `TransformerBlock`: one transformer residual block
- `GPT`: top-level language model
- `KVCacheManager`: active contiguous cache implementation
- `InferenceController`: main generation and profiling orchestrator
- `WorkloadSimulator`: synthetic request generator
- `RequestQueue`: FIFO queue primitive
- `ContinuousBatching`: active request list manager
- `ChunkedPrefill`: chunk generator for long prompts
- `LatencyTracker`: phase/layer timing tracker

### Experimental or placeholder classes

- `PagedKVCache`: placeholder for block/paged cache
- `QuantizedKVCache`: placeholder for compressed KV cache
- `FlashAttention`: alternative attention module not yet wired in
- `FusedMLP`: alternative fused MLP not yet wired in
- `QuantizedLinear`: experimental linear layer for manual INT8 inference

## 8. What Is Actually Implemented vs Planned

### Implemented and used

- toy GPT forward path
- contiguous KV cache
- controller-driven prefill/decode loop
- top-k, top-p, and greedy sampling
- latency and memory profiling
- benchmark result schema
- four benchmark experiment scripts
- plot generation
- controller migration tests

### Implemented but not integrated into main runtime

- flash attention wrapper
- fused attention helpers
- fused layer norm and GELU helpers
- manual model quantization helpers
- workload simulator
- queueing and batching primitives
- chunked prefill utility

### Declared but not implemented

- paged KV cache
- quantized KV cache

## 9. Design Strengths

- The codebase has a clean conceptual split between model, cache, inference loop, profiling, and experiments.
- `InferenceController` gives the repository one central execution path.
- `KVCacheManager.get_for_attention(...)` uses in-place writes and views, which matches the intended optimization goal.
- `build_benchmark_result(...)` provides a consistent result contract across experiments.
- Benchmark scripts are easy to read and modify for further systems exploration.

## 10. Current Technical Gaps and Risks

- The scheduler package is not integrated into the actual generation path.
- `PagedKVCache` and `QuantizedKVCache` are placeholders.
- Flash attention and fused kernels are not used by `GPT`.
- Some docstrings are stale relative to current behavior.
- The notebook duplicates code that now exists in `src/`, creating maintenance drift risk.
- Cache logic assumes a simple batch layout and does not yet manage per-request variable sequence offsets inside a shared batch.
- EOS handling is hardcoded to token ID `0`.
- No explicit causal mask is applied in baseline self-attention.
- Benchmark scripts mostly use synthetic random token inputs rather than realistic tokenized prompts.

## 11. Practical Mental Model for This Repo

The easiest way to understand the project is to think of it in three layers:

### Layer 1: Minimal language model core

- `GPT`
- `TransformerBlock`
- `SelfAttention`
- `MLP`

This is the compute graph.

### Layer 2: Inference-engine mechanics

- `KVCacheManager`
- `InferenceController`
- sampling functions

This is the serving-style runtime logic.

### Layer 3: Measurement and experimentation

- profiling helpers
- workload simulator
- benchmark scripts
- visualization

This is the research and evaluation layer.

That means the project is less a full production server and more a systems-oriented inference research sandbox.

## 12. Recommended Reading Order

If you want to understand the code quickly, read in this order:

1. `src/inference/controller.py`
2. `src/model_core/gpt.py`
3. `src/model_core/transformer.py`
4. `src/model_core/attention.py`
5. `src/kv_cache/contiguous_cache.py`
6. `src/profiling/metrics.py`
7. `benchmarks/exp3_kv_cache.py`
8. `src/optimizations/quantization.py`
9. `src/scheduler/*`

That sequence shows the real runtime first, then the experiments, then the future directions.

## 13. Short Summary

This repository is a prototype LLM inference engine centered on a simple GPT implementation and a contiguous KV cache. The most complete path is:

- `input_ids` -> `InferenceController` -> `GPT` -> transformer blocks -> cache-aware attention -> sampled decode -> benchmark metrics

The strongest implemented idea is the controller plus contiguous cache path. The scheduler, paged cache, quantized KV cache, and some optimization modules represent the next stage of the project rather than finished runtime features.
