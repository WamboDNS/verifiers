# Multi-Agent/Multi-LoRA Support for prime-rl

## Overview

This document describes the design for adding multi-agent training support to prime-rl, enabling different agents to have independent LoRA adapters that are trained and served independently.

## Core Principle

**Adapters stay independent throughout the entire pipeline:**
- Training: Each agent's samples update only their LoRA adapter
- Sync: Each adapter synced separately to vLLM
- Inference: Each agent's requests routed to their specific adapter

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MULTI-AGENT TRAINING                             │
└─────────────────────────────────────────────────────────────────────────┘

1. ENVIRONMENT (verifiers - already implemented)
   MultiAgentEnv.generate() returns:
   ├─ outputs[i]["trajectory"] → standard trajectory steps
   └─ outputs[i]["agent_rollouts"] → per-agent rollouts with meta.lora_id

2. ORCHESTRATOR (prime-rl - needs implementation)
   ├─ Detect MultiAgentEnv via env.get_lora_groups()
   ├─ Extract agent_rollouts from each output
   ├─ Tag each training sample with lora_id
   └─ Return Batch with lora_id per sample

3. TRAINER (prime-rl - needs implementation)
   ├─ Setup multiple LoRA adapters at init
   ├─ Group microbatch samples by lora_id
   ├─ Switch adapter before forward pass
   └─ Each adapter gets gradients only from its samples

4. vLLM SYNC (prime-rl - needs implementation)
   ├─ Sync base model weights (if changed)
   └─ Sync each LoRA adapter separately

5. vLLM INFERENCE (needs implementation)
   ├─ Load multiple LoRA adapters
   └─ Route each request to correct adapter based on agent
```

## Component Changes

### 1. Orchestrator (`orchestrator.py`)

#### New Data Structures

```python
class Microbatch(BaseModel):
    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    items: int
    # NEW: per-sample LoRA assignment
    lora_ids: list[int | None] = Field(default_factory=list)


class Batch(BaseModel):
    # ... existing fields ...
    # NEW: LoRA distribution for logging
    lora_id_counts: dict[str, int] = Field(default_factory=dict)
```

#### Detection Logic

```python
def __init__(self, env, ...):
    # Detect multi-agent environment
    self.is_multi_agent = hasattr(env, 'get_lora_groups')
    if self.is_multi_agent:
        self.lora_groups = env.get_lora_groups()
        self.trainable_agents = set(env.get_trainable_agents())
    else:
        self.lora_groups = {}
        self.trainable_agents = set()
```

#### Batch Generation

```python
async def generate_batch(self, batch_id: int) -> Batch:
    env_results = await self.env.generate(
        repeated_ds,
        client=self.client,
        model=self.model_name,
        sampling_args=self.sampling_args,
        max_concurrent=self.max_concurrent,
        state_columns=["trajectory", "agent_rollouts"] if self.is_multi_agent else ["trajectory"],
    )

    if self.is_multi_agent:
        return self._process_multi_agent_batch(batch_id, env_results, wall_clock_s)
    else:
        return self._process_standard_batch(batch_id, env_results, wall_clock_s)


def _process_multi_agent_batch(self, batch_id, env_results, wall_clock_s) -> Batch:
    """Process batch from MultiAgentEnv with per-agent rollouts."""
    prompt_ids, prompt_mask = [], []
    completion_ids, completion_mask = [], []
    completion_logprobs, advantages = [], []
    lora_ids = []  # Track lora_id per sample

    per_agent_rewards: dict[str, list[float]] = {}

    for output in env_results["outputs"]:
        agent_rollouts = output.get("agent_rollouts", [])

        for agent_rollout in agent_rollouts:
            meta = agent_rollout.get("meta", {})
            agent_id = meta.get("agent_id")
            trainable = meta.get("trainable", True)
            lora_id = meta.get("lora_id")

            if not trainable:
                continue

            # Track per-agent rewards
            agent_reward = agent_rollout.get("total_reward", 0.0)
            if agent_id not in per_agent_rewards:
                per_agent_rewards[agent_id] = []
            per_agent_rewards[agent_id].append(agent_reward)

            # Process each step
            for step in agent_rollout.get("steps", []):
                tokens = step.get("tokens")
                if tokens is None:
                    continue

                prompt_ids.append(tokens["prompt_ids"])
                prompt_mask.append(tokens["prompt_mask"])
                completion_ids.append(tokens["completion_ids"])
                completion_mask.append(tokens["completion_mask"])
                completion_logprobs.append(tokens["completion_logprobs"])
                advantages.append(step.get("advantage", 0.0) or 0.0)
                lora_ids.append(lora_id)

    # Build microbatches with lora_ids included
    # ... (same distribution logic, include lora_ids in Microbatch)
```

### 2. Trainer (`trainer.py`)

#### Configuration

```python
@dataclass
class RLConfig(TrainingArguments):
    # ... existing fields ...

    # Multi-LoRA settings
    multi_lora: bool = False
    lora_ids: list[int] = field(default_factory=list)  # Auto-detect if empty
```

#### Initialization

```python
def __init__(self, model, env, args, ...):
    # Detect multi-agent environment
    self.is_multi_agent = hasattr(env, 'get_lora_groups')
    self.lora_groups = {}
    self.lora_ids = []

    if self.is_multi_agent:
        self.lora_groups = env.get_lora_groups()
        self.lora_ids = sorted([lid for lid in self.lora_groups.keys() if lid is not None])
        self.has_base_model_agents = None in self.lora_groups

    # Setup model with LoRA
    if args.use_lora:
        if self.lora_ids and args.multi_lora:
            model = self._setup_multi_lora(model, args)
        else:
            model = prepare_peft_model(model, args.lora_config, args)


def _setup_multi_lora(self, model, args):
    """Setup multiple independent LoRA adapters."""
    # Create first adapter
    model = prepare_peft_model(model, args.lora_config, args)

    # Rename default adapter to lora_{first_id}
    first_id = self.lora_ids[0]
    adapter_name = f"lora_{first_id}"
    model.peft_config[adapter_name] = model.peft_config.pop("default")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            module.lora_A[adapter_name] = module.lora_A.pop("default")
            module.lora_B[adapter_name] = module.lora_B.pop("default")

    # Create additional adapters
    for lora_id in self.lora_ids[1:]:
        adapter_name = f"lora_{lora_id}"
        model.add_adapter(adapter_name, args.lora_config)

    model.set_adapter(f"lora_{self.lora_ids[0]}")
    return model
```

#### Training Step

```python
def training_step(self, model, *args, **kwargs):
    # ... existing setup ...

    is_peft = is_peft_model(self.model)

    for microbatch in local_microbatches:
        if self.is_multi_agent and hasattr(microbatch, 'lora_ids') and microbatch.lora_ids:
            # Group samples by lora_id
            lora_groups = self._group_by_lora(microbatch)

            for lora_id, group_data in lora_groups.items():
                if not group_data["input_ids"]:
                    continue

                # Switch to correct adapter
                if is_peft and lora_id is not None:
                    self.model.set_adapter(f"lora_{lora_id}")

                # Forward/backward for this adapter's samples only
                loss, summaries = self._forward_backward(
                    model, group_data, inv_tokens_per_rank
                )
                total_loss += loss
        else:
            # Standard single-adapter processing
            loss, summaries = self._forward_backward(
                model, microbatch, inv_tokens_per_rank
            )
            total_loss += loss

    # Single optimizer step updates all adapters that received gradients
    return total_loss


def _group_by_lora(self, microbatch) -> dict[int | None, dict]:
    """Group microbatch samples by their lora_id."""
    groups = defaultdict(lambda: {
        "input_ids": [], "loss_mask": [],
        "sampling_logprobs": [], "advantages": []
    })

    for i, lora_id in enumerate(microbatch.lora_ids):
        groups[lora_id]["input_ids"].append(microbatch.input_ids[i])
        groups[lora_id]["loss_mask"].append(microbatch.loss_mask[i])
        groups[lora_id]["sampling_logprobs"].append(microbatch.sampling_logprobs[i])
        groups[lora_id]["advantages"].append(microbatch.advantages[i])

    return dict(groups)
```

### 3. vLLM Weight Sync

Each adapter is synced independently - no merging.

```python
def update_vllm(self):
    # ... wait for generation, setup gather context ...

    if is_peft_model(self.model) and self.lora_ids:
        # Multi-LoRA: sync each adapter separately
        with gather_if_zero3(list(self.model.parameters())):
            for lora_id in self.lora_ids:
                adapter_name = f"lora_{lora_id}"
                self.model.set_adapter(adapter_name)

                # Get adapter-specific weights
                adapter_state = {}
                for name, param in self.model.named_parameters():
                    if adapter_name in name or "lora_" in name:
                        # Extract clean name for vLLM
                        clean_name = self._clean_adapter_param_name(name, adapter_name)
                        adapter_state[clean_name] = param.data

                # Sync this adapter to vLLM
                if self.client:
                    self.client.update_lora_adapter(lora_id, adapter_state)

    elif is_peft_model(self.model):
        # Single LoRA: existing merge/sync/unmerge logic
        with gather_if_zero3(list(self.model.parameters())):
            self.model.merge_adapter()
            for name, param in self.model.named_parameters():
                # ... existing sync logic ...
            self.model.unmerge_adapter()

    else:
        # No LoRA: sync base model directly
        for name, param in self.model.named_parameters():
            with gather_if_zero3([param]):
                if self.client:
                    self.client.update_named_param(name, param.data)
```

### 4. vLLM Server Changes

#### Loading Multiple Adapters

```python
# vLLM server startup with multiple LoRA adapters
# Option A: Pre-register adapters
llm = LLM(
    model="base_model",
    enable_lora=True,
    max_loras=4,  # Max concurrent adapters
    max_lora_rank=64,
)

# Option B: Dynamic adapter loading via API
@app.post("/load_lora_adapter")
async def load_adapter(lora_id: int, weights: dict):
    # Store adapter weights for this lora_id
    ...

@app.post("/update_lora_adapter")
async def update_adapter(lora_id: int, weights: dict):
    # Update specific adapter weights
    ...
```

#### Routing Requests to Correct Adapter

```python
# In environment or orchestrator, when making inference requests:
response = await client.chat.completions.create(
    model=model_name,
    messages=messages,
    extra_body={
        "lora_request": {
            "lora_name": f"lora_{agent.lora_id}",
            "lora_int_id": agent.lora_id,
        }
    } if agent.lora_id is not None else {},
)
```

### 5. Environment Changes (verifiers)

The environment needs to pass `lora_id` when making inference requests so vLLM uses the correct adapter.

```python
# In MultiAgentEnv._get_agent_response():
async def _get_agent_response(self, agent: Agent, state: State):
    prompt = agent.get_prompt()
    client = agent.client or state["client"]
    model = agent.model or state["model"]
    sampling_args = {**state.get("sampling_args", {}), **agent.sampling_args}

    # Add lora_id to request if agent has one
    extra_body = {}
    if agent.lora_id is not None:
        extra_body["lora_request"] = {
            "lora_name": f"lora_{agent.lora_id}",
            "lora_int_id": agent.lora_id,
        }

    response = await self.get_model_response(
        state=state,
        prompt=prompt,
        client=client,
        model=model,
        sampling_args=sampling_args,
        extra_body=extra_body,
    )
    return response
```

## Interface Summary

### Environment → Orchestrator

```python
# MultiAgentEnv provides:
env.get_lora_groups() -> dict[int | None, list[str]]
env.get_trainable_agents() -> list[str]

# Rollout output includes:
output["agent_rollouts"] = [
    {
        "steps": [TrajectoryStep, ...],
        "total_reward": float,
        "meta": {
            "agent_id": str,
            "trainable": bool,
            "lora_id": int | None,
        }
    },
    ...
]
```

### Orchestrator → Trainer

```python
microbatch.lora_ids: list[int | None]  # per-sample LoRA assignment
batch.lora_id_counts: dict[str, int]   # sample distribution
batch.metrics_dict["reward/{agent_id}"] # per-agent metrics
```

### Trainer → vLLM

```python
client.update_lora_adapter(lora_id: int, weights: dict[str, Tensor])
```

### Environment → vLLM (inference)

```python
extra_body={"lora_request": {"lora_name": f"lora_{lora_id}", "lora_int_id": lora_id}}
```

## Implementation Status

### Completed

**verifiers (environment layer):**
- `MultiAgentEnv`: Multi-agent rollouts with per-agent trajectory collection
- `Agent`/`AgentConfig`: Agent abstraction with `lora_id` and `trainable` fields
- `TraceCollector`: Per-agent trajectory collection with `extract_rollouts()`
- Inference routing: `lora_request` in extra_body for vLLM multi-LoRA serving

**prime-rl (training layer):**
- `TrainingSample.lora_id`: Per-sample LoRA assignment
- `process_multi_agent_rollout()`: Extracts agent rollouts with lora_id tagging
- `SinglePacker`: Routes multi-agent samples to correct LoRA adapters
- Validation: Fails fast if lora_id exceeds configured max_runs

### Configuration Requirement

For multi-agent training, set `max_runs` in trainer config to be >= max(lora_id) + 1:

```toml
# Example: 2 agents with lora_id 0 and 1
[trainer]
max_runs = 2
```

### Remaining Work

1. **vLLM multi-adapter serving**: Load and serve multiple LoRA adapters
2. **Per-adapter weight sync**: Sync each adapter separately to vLLM
3. **Per-agent metrics**: Track and log rewards per agent_id

## Metrics & Logging

```python
# Per-agent rewards
metrics_dict["reward"] = overall_mean
metrics_dict["reward/{agent_id}"] = agent_specific_mean

# LoRA distribution
metrics_dict["samples/lora_0"] = count_lora_0
metrics_dict["samples/lora_1"] = count_lora_1
metrics_dict["samples/base"] = count_base_model

# wandb table includes per-agent reward columns
```

## Open Questions

1. **Shared vs independent optimizer state**: Currently all adapters share one optimizer. Could have per-adapter optimizers with different LRs.

2. **Adapter initialization**: Should new adapters start from scratch or copy from another? (Current: from scratch via `add_adapter()`)

3. **Dynamic adapter count**: What if agents are added/removed during training? (Current: fixed at init)
