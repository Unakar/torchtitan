# Agent Notes

## TorchTitan Architecture
- `torchtitan/train.py` defines `Trainer` orchestrating the job: it loads `JobConfig`, initializes device/distributed meshes, and builds tokenizer, dataloader, model parts, loss, optimizers, LR schedulers, and metrics via the registered `TrainSpec`.
- Config flow comes through `ConfigManager` and dataclasses in `torchtitan/config/job_config.py`; `TrainSpec` instances (registered in model/experiment modules) hold builder callables for each component of the training stack.
- Model-specific logic lives under `torchtitan/models` and `torchtitan/experiments`, each registering a `TrainSpec` with appropriate parallelization/pipelining helpers.
- Core component builders under `torchtitan/components` implement reusable pieces: `optimizer.py` supplies optimizer containers, TorchFT wrappers, and optimizer-in-backward support; other modules cover dataloaders, LR schedulers, metrics, etc.
- Distributed topology is handled through `torchtitan/distributed/ParallelDims`, which creates the `DeviceMesh`/process groups consumed by models, optimizers, and optional MoE load-balancing hooks.

## Current Optimizer Path
- `torchtitan/components/optimizer.py` exposes `build_optimizers` (and a MoE-aware variant) that return optimizer containers per model part.
- When `optimizer.name == "muon"`, the builder now delegates to `experiments/muon/titon_muon.build_muon_optimizers`, returning a `MuonOptimizersContainer` that wraps one Muon instance per model part.
- For other names the original Adam/AdamW pathway is preserved, including optional TorchFT or optimizer-in-backward support.
- Train specs continue to call these builders directly, so selecting Muon in config switches behaviour without further changes in model registries.

## Muon Implementation Snapshot
- `experiments/muon/muon.py` provides the Muon optimizer with async batched updates, DTensor-awareness, and optional Triton Newton-Schulz kernels; it also drives AdamW/Lion style updates for scalar groups.
- `opt_utils.py` offers DTensor<->local conversions, parameter batching, and an async runtime; `scalar_opts.py` defines compiled AdamW update helpers; `newton_schulz_triton.py` hosts the Triton kernels.
- `parameter_classification.py` now produces Muon-focused parameter groups (matrix, scalar, embedding, head, routing, expert) and returns summary stats so callers can log once per run.
- `titon_muon.py` defines `MuonOptimizersContainer`, a `torch.optim.Optimizer`/`Stateful` hybrid that instantiates one Muon per model part and feeds TorchTitan's checkpoint utilities.
- `muon_test.toml` exercises the new config pathway where `optimizer.name = "muon"` plus Muon-specific fields map directly to the container.

## Integration Gaps / Considerations
- TorchFT, optimizer-in-backward, and MoE load-balancing are still wired only for Adam/AdamW; Muon should explicitly guard or extend those paths once requirements are clear.
- Muon parameter classification emits verbose logging; we may want to gate verbosity or hook into the core metrics processor later.
- State-dict compatibility with existing TorchTitan checkpoints requires validation (save/load cycles, reshaping under different parallel meshes).
- Additional polish (e.g., exposing expert routing overrides, gradient sync modes) may be needed before promoting Muon outside experiments.

## Proposed Plan
1. Validate the end-to-end Muon path (e.g., run `muon_test.toml`, ensure checkpoint save/load and DTensor sharding behave as expected).
2. Decide on TorchFT/optimizer-in-backward handling for Muon (explicitly disable with clear errors or extend support).
3. Tighten logging/telemetry so Muon stats funnel through core metrics rather than ad-hoc prints.
4. Circle back on documentation: surface Muon configuration options in user-facing docs once behaviour is verified.

# Muon Integration

- torchtitan/components/optimizer.py:265 now short-circuits to the Muon builder when optimizer.name == "muon", keeping the Adam/AdamW flow untouched for everything else.
- torchtitan/config/job_config.py:138 extends the Optimizer dataclass with Muon-specific hyperparameters and LR scaling knobs so configs like muon_test.toml parse cleanly.
- torchtitan/experiments/muon/titon_muon.py:33 replaces the placeholder container with a torch.optim.Optimizer/Stateful wrapper that spins up one Muon instance per model part,
  reuses the shared device mesh, and emits a single classification summary.
- torchtitan/experiments/muon/parameter_classification.py:51 drops the Dion logic in favour of Muon-only grouping, returning both param groups and stats so callers can log once
  after building optimizers.
- agent.md aligns with the new architecture and follow-up work.

Notes

- The Muon experiment files (torchtitan/experiments/muon/*) and agent.md are still untracked—git add them when you’re ready.
- No tests or training runs executed: the workspace is read-only and the Muon path needs a writable cache for compiled kernels.

Next Steps

1. Run the muon_test.toml config end-to-end to confirm build, train, and checkpoint save/load on your target mesh.
2. Decide how Muon should interact with TorchFT and optimizer-in-backward (guard with clear errors or add support).
3. If Muon will ship broadly, add user-facing docs and tighten logging/metrics routing.