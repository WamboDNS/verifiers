import asyncio
import logging
import queue
import threading
import time
from typing import Any

import httpx
import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from verifiers import Environment
from verifiers.types import Messages


class Microbatch(BaseModel):
    """Microbatch for batch generation."""

    input_ids: list[list[int]]
    loss_mask: list[list[int]]
    sampling_logprobs: list[list[float]]
    advantages: list[list[float]]
    items: int
    # Per-sample LoRA assignment for multi-agent training
    lora_ids: list[int | None] = Field(default_factory=list)


class Batch(BaseModel):
    """Result from batch generation."""

    batch_id: int
    microbatches: list[list[Microbatch]]
    items_per_process: list[int]
    global_item_count: int
    # LoRA distribution for multi-agent training
    lora_id_counts: dict[str, int] = Field(default_factory=dict)
    # logging
    generation_time: float = 0.0
    prompts: list[Any] = Field(default_factory=list)
    completions: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    metrics_dict: dict[str, float] = Field(default_factory=dict)
    rewards_dict: dict[str, list[float]] = Field(default_factory=dict)


class Orchestrator:
    """
    Manages asynchronous batch generation in parallel with RL training.
    """

    def __init__(
        self,
        env: Environment,
        client_base_url: str,
        client_api_key: str,
        client_limit: int,
        client_timeout: float,
        model_name: str,
        sampling_args: dict[str, Any],
        rollouts_per_example: int,
        batch_size: int,
        micro_batch_size: int,
        num_processes: int,
        generation_timeout: float,
        processing_class: PreTrainedTokenizerBase,
        mask_env_responses: bool,
        max_seq_len: int,
        max_prompt_len: int,
        mask_truncated_completions: bool,
        zero_truncated_completions: bool,
        max_concurrent: int,
    ):
        self.env = env
        self.client_base_url = client_base_url
        self.client_api_key = client_api_key
        self.client_limit = client_limit
        self.client_timeout = client_timeout
        self.client = None  # created in worker thread
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.rollouts_per_example = rollouts_per_example
        self.prompts_per_batch = batch_size // rollouts_per_example
        self.micro_batch_size = micro_batch_size
        self.num_processes = num_processes
        self.generation_timeout = generation_timeout
        self.processing_class = processing_class
        self.mask_env_responses = mask_env_responses
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.mask_truncated_completions = mask_truncated_completions
        self.zero_truncated_completions = zero_truncated_completions
        self.max_concurrent = max_concurrent

        # queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_generating = False
        self.completed_batches = {}

        self.worker_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.is_generating = False
        self.worker_loop = None

        # Detect multi-agent environment
        self.is_multi_agent = hasattr(env, "get_lora_groups")
        if self.is_multi_agent:
            self.lora_groups = env.get_lora_groups()
            self.trainable_agents = set(env.get_trainable_agents())
            self.logger.info(
                f"Multi-agent environment detected: "
                f"{len(self.trainable_agents)} trainable agents, "
                f"{len(self.lora_groups)} LoRA groups"
            )
            for lora_id, agents in self.lora_groups.items():
                lora_name = f"lora_{lora_id}" if lora_id is not None else "base"
                self.logger.info(f"  {lora_name}: {agents}")
        else:
            self.lora_groups = {}
            self.trainable_agents = set()

        max_length = self.max_prompt_len

        def filter_by_prompt_length(example, processing_class):
            prompt = example["prompt"]
            if isinstance(prompt, list):
                prompt_text = processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = prompt
            prompt_ids = processing_class.encode(prompt_text)
            return len(prompt_ids) <= max_length

        env.dataset = env.get_dataset().filter(
            filter_by_prompt_length,
            fn_kwargs={"processing_class": processing_class},
        )

    def get_dataset_slice(self, batch_id: int) -> Dataset:
        """Get dataset slice for a given batch id"""
        num_rows = self.prompts_per_batch
        dataset = self.env.get_dataset()
        total_rows = len(dataset)
        if total_rows == 0:
            raise ValueError("Environment dataset is empty")
        offset = (batch_id * num_rows) % total_rows
        indices = [(offset + i) % total_rows for i in range(num_rows)]
        return dataset.select(indices)

    def start(self):
        """Start the async generation worker thread"""
        self.worker_thread = threading.Thread(
            target=self.generation_worker, daemon=True, name="BatchGenerator"
        )
        self.worker_thread.start()

    def stop(self):
        """Stop the async generation worker thread"""
        self.stop_event.set()
        self.request_queue.put(None)  # poison pill
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)

    def submit_batch(self, batch_id: int):
        self.request_queue.put(batch_id)

    def get_batch(self, batch_id: int) -> Batch:
        """
        Get a completed batch result. Blocks until the batch is ready.

        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait

        Returns:
            BatchResult: The completed batch result

        Raises:
            TimeoutError: batch doesn't complete within timeout
            RuntimeError: generation failed
        """
        timeout = self.generation_timeout
        start_time = time.time()
        while True:
            if batch_id in self.completed_batches:
                return self.completed_batches.pop(batch_id)
            try:
                result = self.result_queue.get(timeout=0.1)
                self.completed_batches[result.batch_id] = result
                if result.batch_id == batch_id:
                    return self.completed_batches.pop(batch_id)
            except queue.Empty:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout}s")

    def generation_worker(self):
        """Worker thread that processes generation requests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loop = loop
        self.client = AsyncOpenAI(
            base_url=self.client_base_url,
            api_key=self.client_api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.client_limit),
                timeout=self.client_timeout,
            ),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    batch_id = self.request_queue.get(timeout=0.1)
                    if batch_id is None:  # poison pill
                        break
                    result = loop.run_until_complete(self.generate_batch(batch_id))
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in generation worker: {e}")
                    raise e
        finally:
            loop.run_until_complete(self.client.close())
            loop.close()
            asyncio.set_event_loop(None)

    async def generate_batch(self, batch_id: int) -> Batch:
        """Generate a single batch asynchronously."""
        self.is_generating = True
        assert self.client is not None
        start_time = time.time()
        batch_ds = self.get_dataset_slice(batch_id)
        repeated_ds = batch_ds.repeat(self.rollouts_per_example)

        # Request agent_rollouts for multi-agent environments
        state_columns = ["trajectory"]
        if self.is_multi_agent:
            state_columns.append("agent_rollouts")

        env_results = await self.env.generate(
            repeated_ds,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            max_concurrent=self.max_concurrent,
            state_columns=state_columns,
        )
        self.is_generating = False
        wall_clock_s = time.time() - start_time

        outputs = env_results["outputs"]

        if self.is_multi_agent:
            return self._process_multi_agent_batch(
                batch_id, outputs, wall_clock_s
            )
        else:
            return self._process_standard_batch(
                batch_id, outputs, wall_clock_s
            )

    def _process_standard_batch(
        self,
        batch_id: int,
        outputs: list[dict[str, Any]],
        wall_clock_s: float,
    ) -> Batch:
        """Process batch from standard single-agent environment."""
        prompt_ids: list[list[int]] = []
        prompt_mask: list[list[int]] = []
        completion_ids: list[list[int]] = []
        completion_mask: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        prompts: list[Messages] = []
        completions: list[Messages] = []
        rewards: list[float] = []
        metrics: dict[str, list[float]] = {}
        advantages: list[float] = []

        for output in outputs:
            trajectory = output["trajectory"]
            for step in trajectory:
                tokens = step["tokens"]
                if tokens is None:
                    continue
                prompt_ids.append(tokens["prompt_ids"])
                prompt_mask.append(tokens["prompt_mask"])
                completion_ids.append(tokens["completion_ids"])
                completion_mask.append(tokens["completion_mask"])
                completion_logprobs.append(tokens["completion_logprobs"])
                advantages.append(step["advantage"])
            prompts.append(output["prompt"])
            completions.append(output["completion"])
            rewards.append(output["reward"])
            for k, v in output["metrics"].items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        rewards_dict, metrics_dict = self._compute_metrics(
            rewards, metrics, advantages, completion_ids, completion_mask, outputs
        )
        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)
        errors = [output.get("error") for output in outputs]

        # No lora_ids for standard batch
        lora_ids: list[int | None] = [None] * len(advantages)

        microbatches, items_per_process = self._build_microbatches(
            prompt_ids, prompt_mask, completion_ids, completion_mask,
            completion_logprobs, advantages, lora_ids
        )

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=sum(items_per_process),
            lora_id_counts={},
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=completions,
            prompts=prompts,
            errors=errors,
            metrics_dict=metrics_dict,
        )

    def _process_multi_agent_batch(
        self,
        batch_id: int,
        outputs: list[dict[str, Any]],
        wall_clock_s: float,
    ) -> Batch:
        """Process batch from multi-agent environment with per-agent rollouts."""
        prompt_ids: list[list[int]] = []
        prompt_mask: list[list[int]] = []
        completion_ids: list[list[int]] = []
        completion_mask: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        advantages: list[float] = []
        lora_ids: list[int | None] = []

        # Rollout-level data for logging
        prompts: list[Messages] = []
        completions: list[Messages] = []
        rewards: list[float] = []
        metrics: dict[str, list[float]] = {}
        per_agent_rewards: dict[str, list[float]] = {}

        for output in outputs:
            # Log rollout-level data
            prompts.append(output["prompt"])
            completions.append(output["completion"])
            rewards.append(output["reward"])

            for k, v in output.get("metrics", {}).items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

            # Extract per-agent rollouts for training
            agent_rollouts = output.get("agent_rollouts", [])

            for agent_rollout in agent_rollouts:
                meta = agent_rollout.get("meta", {})
                agent_id = meta.get("agent_id", "unknown")
                trainable = meta.get("trainable", True)
                lora_id = meta.get("lora_id")

                if not trainable:
                    continue

                # Track per-agent rewards
                agent_reward = agent_rollout.get("total_reward", 0.0)
                if agent_id not in per_agent_rewards:
                    per_agent_rewards[agent_id] = []
                per_agent_rewards[agent_id].append(agent_reward)

                # Process each step in the agent's trajectory
                steps = agent_rollout.get("steps", [])
                for step in steps:
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

        # Build rewards_dict with per-agent rewards
        rewards_dict: dict[str, list[float]] = {"reward": rewards}
        for k in metrics:
            rewards_dict[k] = metrics[k]
        for agent_id, agent_rewards in per_agent_rewards.items():
            rewards_dict[f"reward/{agent_id}"] = agent_rewards

        # Compute metrics
        metrics_dict: dict[str, float] = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        # Per-agent reward metrics
        for agent_id, agent_rewards in per_agent_rewards.items():
            if agent_rewards:
                arr = np.asarray(agent_rewards, dtype=np.float32)
                metrics_dict[f"reward/{agent_id}"] = float(arr.mean())

        # Completion length metrics
        completion_lengths = [len(ids) for ids in completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

        # LoRA distribution metrics
        lora_id_counts: dict[str, int] = {}
        for lid in lora_ids:
            key = str(lid) if lid is not None else "base"
            lora_id_counts[key] = lora_id_counts.get(key, 0) + 1

        for lora_key, count in lora_id_counts.items():
            metrics_dict[f"samples/lora_{lora_key}"] = float(count)

        metrics_dict["wall_clock/generate_s"] = float(wall_clock_s)
        errors = [output.get("error") for output in outputs]

        # Log generation summary
        self.logger.info(
            f"Batch {batch_id}: {len(outputs)} rollouts, "
            f"{len(advantages)} training samples in {wall_clock_s:.1f}s"
        )
        for lora_key, count in lora_id_counts.items():
            self.logger.debug(f"  lora_{lora_key}: {count} samples")

        if not advantages:
            # No trainable samples
            return Batch(
                batch_id=batch_id,
                microbatches=[[] for _ in range(self.num_processes)],
                items_per_process=[0] * self.num_processes,
                global_item_count=0,
                lora_id_counts=lora_id_counts,
                generation_time=wall_clock_s,
                rewards_dict=rewards_dict,
                completions=completions,
                prompts=prompts,
                errors=errors,
                metrics_dict=metrics_dict,
            )

        microbatches, items_per_process = self._build_microbatches(
            prompt_ids, prompt_mask, completion_ids, completion_mask,
            completion_logprobs, advantages, lora_ids
        )

        return Batch(
            batch_id=batch_id,
            microbatches=microbatches,
            items_per_process=items_per_process,
            global_item_count=sum(items_per_process),
            lora_id_counts=lora_id_counts,
            generation_time=wall_clock_s,
            rewards_dict=rewards_dict,
            completions=completions,
            prompts=prompts,
            errors=errors,
            metrics_dict=metrics_dict,
        )

    def _compute_metrics(
        self,
        rewards: list[float],
        metrics: dict[str, list[float]],
        advantages: list[float],
        completion_ids: list[list[int]],
        completion_mask: list[list[int]],
        outputs: list[dict[str, Any]],
    ) -> tuple[dict[str, list[float]], dict[str, float]]:
        """Compute metrics from batch data."""
        rewards_dict: dict[str, list[float]] = {"reward": rewards}
        for k in metrics:
            rewards_dict[k] = metrics[k]

        metrics_dict: dict[str, float] = {}
        if rewards:
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            metrics_dict["reward"] = float(rewards_arr.mean())
            metrics_dict["reward/std"] = float(rewards_arr.std())

        if advantages:
            adv_arr = np.asarray(advantages, dtype=np.float32)
            metrics_dict["advantage/absmean"] = float(np.abs(adv_arr).mean())

        for reward_name, values in rewards_dict.items():
            if reward_name == "reward":
                continue
            if len(values) == 0:
                continue
            reward_values = np.asarray(values, dtype=np.float32)
            metrics_dict[f"reward/{reward_name}"] = float(reward_values.mean())

        completion_lengths = [len(ids) for ids in completion_ids]
        if completion_lengths:
            completion_lengths_arr = np.asarray(completion_lengths, dtype=np.float32)
            metrics_dict["tokens/completion"] = float(completion_lengths_arr.mean())

            completion_mask_lengths = np.asarray(
                [sum(mask) for mask in completion_mask],
                dtype=np.float32,
            )
            valid_tokens = completion_mask_lengths.sum()
            total_tokens = completion_lengths_arr.sum()
            if total_tokens > 0:
                masked_fraction = 1.0 - (valid_tokens / total_tokens)
                metrics_dict["tokens/masked_fraction"] = float(masked_fraction)

        generation_ms: list[float] = []
        scoring_ms: list[float] = []
        total_ms: list[float] = []
        for output in outputs:
            timing = output.get("timing", {})
            if "generation_ms" in timing:
                generation_ms.append(float(timing["generation_ms"]))
            if "scoring_ms" in timing:
                scoring_ms.append(float(timing["scoring_ms"]))
            if "total_ms" in timing:
                total_ms.append(float(timing["total_ms"]))

        if generation_ms:
            metrics_dict["timing/generation_ms"] = float(np.mean(generation_ms))
        if scoring_ms:
            metrics_dict["timing/scoring_ms"] = float(np.mean(scoring_ms))
        if total_ms:
            metrics_dict["timing/total_ms"] = float(np.mean(total_ms))

        return rewards_dict, metrics_dict

    def _build_microbatches(
        self,
        prompt_ids: list[list[int]],
        prompt_mask: list[list[int]],
        completion_ids: list[list[int]],
        completion_mask: list[list[int]],
        completion_logprobs: list[list[float]],
        advantages: list[float],
        lora_ids: list[int | None],
    ) -> tuple[list[list[Microbatch]], list[int]]:
        """Build per-process microbatches from training data."""
        N = len(advantages)
        per_proc = N // self.num_processes
        microbatches: list[list[Microbatch]] = []
        items_per_process: list[int] = []

        for proc in range(self.num_processes):
            ps = proc * per_proc
            pe = ps + per_proc
            proc_mbs: list[Microbatch] = []
            proc_item_total = 0

            for s in range(ps, pe, self.micro_batch_size):
                e = min(s + self.micro_batch_size, pe)
                ids_chunk = [prompt_ids[i] + completion_ids[i] for i in range(s, e)]
                mask_chunk = [prompt_mask[i] + completion_mask[i] for i in range(s, e)]
                logprobs_chunk = [
                    [0.0] * len(prompt_mask[i]) + completion_logprobs[i]
                    for i in range(s, e)
                ]
                lengths = [len(mask) for mask in mask_chunk]
                adv_chunk = [
                    [advantages[i]] * lengths[idx]
                    for idx, i in enumerate(range(s, e))
                ]
                lora_chunk = [lora_ids[i] for i in range(s, e)]
                mb_items = sum(sum(mask) for mask in mask_chunk)

                microbatch = Microbatch(
                    input_ids=ids_chunk,
                    loss_mask=mask_chunk,
                    sampling_logprobs=logprobs_chunk,
                    advantages=adv_chunk,
                    items=mb_items,
                    lora_ids=lora_chunk,
                )
                proc_item_total += mb_items
                proc_mbs.append(microbatch)

            microbatches.append(proc_mbs)
            items_per_process.append(proc_item_total)

        return microbatches, items_per_process
