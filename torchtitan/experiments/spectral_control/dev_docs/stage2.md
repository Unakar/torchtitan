# Lipschitz-transformer
在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/dev/repos/lipschitz-transformers里，

论文可以阅读https://arxiv.org/abs/2507.13338，理解其大意， 在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/dev/instructions/spectral_control_gpt5.md有一份gpt5生成的简介(可能未必正确)

该论文和对应的代码是我们此次项目的主要灵感来源，可以说是我们torchtitan开发就是把这篇文章的思想迁移过来。请仔细阅读分析！！！不要偷懒只看一部分

# Torchtitan dev plan
## you may want to know
1. optimizer定义在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/torchtitan/torchtitan/components/optimizer.py里
2. 我们想尝试的实验可在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/torchtitan/torchtitan/experiments/spectral_control进行(目前是空文件夹)
3. 现有的model arch以torchtitan/torchtitan/experiments/qwen3为模板，可以先从0.6B model (torchtitan/torchtitan/experiments/qwen3/train_configs/qwen3_0.6b.toml)开始
qwen3下面还包括了文件夹torchtitan/torchtitan/experiments/qwen3/infra和torchtitan/torchtitan/experiments/qwen3/model，注意读取(为了方便开发，你可以直接复制部分文件到/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/torchtitan/torchtitan/experiments/spectral_control下面)
4. 请编写代码对qwen3实现spectral_hammer, spectral_weight_decay, spectral_normalize等等，可利用的函数在torchtitan/torchtitan/spectral_utils.py。在编写代码前，请仔细分析原理，确保理论无误
5. 在你coding前，记得把思考和计划放在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/agent.md里，作为你agent思考，存放草稿和有价值信息，方便人类阅读监管(或者有利于我后续对你行为进行改正的地方)
6. 你更新代码主要改动是什么，这么做的动机和原理是什么，也请在/home/t2vg-a100-G2-1/a_xietian/dev/SpectralNorm-GPT/agent.md里详细说明

# gpt5 answer
> I’ve read your stage2 plan and implemented a spectral-control experiment on top of Qwen3 in TorchTitan, including my thought process and rationale.

  What I added

  - agent.md:1
      - My plan, assumptions, design choices, and how the projection is integrated. Includes next-step ideas.
  - torchtitan/torchtitan/spectral_utils.py (from stage1):1
      - Ready-to-use ops: orthogonalize, hard_cap, soft_cap, pure_svd, power_iterate, spectral_hammer, spectral_weight_decay, spectral_normalize, soft_cap_coupling.
  - New experiment: torchtitan/torchtitan/experiments/spectral_control/init.py:1
      - Registers TrainSpec name "spectral_control".
      - Reuses Qwen3 model + parallelize, but overrides build_optimizers_fn to attach a spectral post-step hook.
  - Post-step projector: torchtitan/torchtitan/experiments/spectral_control/projector.py:1
      - Traverses model parts and applies projection to nn.Linear weights after each optimizer step.
      - Supports methods: soft_cap, hard_cap, normalize, hammer, weight_decay, svd_cap.
      - Scale handling: for (out, in) weights uses scale = sqrt(out/in); operates in the scaled space and maps back. Optional sensitive_to_wmax for w_max semantics.
      - Filtering: excludes embeddings and LM head by default; supports include/exclude FQNs.
  - Config extension (optional, for tuning): torchtitan/torchtitan/experiments/spectral_control/custom_args.py:1
      - Adds a spectral section to JobConfig via experimental.custom_args_module.
      - Fields: enable, method, w_max, spectral_wd, alpha, use_coupling, max_update_norm, sensitive_to_wmax, include_fqns, exclude_fqns.
  - Example config: torchtitan/torchtitan/experiments/spectral_control/train_configs/qwen3_0.6b_spectral.toml:1
      - Sets model.name = "spectral_control", flavor "0.6B".
      - Adds [experimental].custom_args_module and a [spectral] block with example values.

  How it works

  - Trainer builds optimizers per usual; our TrainSpec replaces build_optimizers_fn to attach a step post-hook.
  - The hook projects nn.Linear weights per the selected method after each optimizer step.
  - Defaults: in spectral_control experiment, projection is enabled with soft_cap; it excludes token embeddings and final LM head by default.
  - Coupling: for soft_cap, alpha is either a fixed value or computed with soft_cap_coupling(w_max, wd=0, max_update_norm≈lr). You can override via the [spectral] section.

  Usage

  - Quick start with defaults (soft_cap) using the example config:
      - Set --job.config-file to torchtitan/torchtitan/experiments/spectral_control/train_configs/qwen3_0.6b_spectral.toml
  - To customize spectral behavior via config:
      - Keep [experimental].custom_args_module = "torchtitan.experiments.spectral_control.custom_args".
      - Adjust the [spectral] block (e.g., method="normalize", w_max=1.0, spectral_wd=0.1, include/exclude FQNs).

  Notes and rationale

  - Projection timing: Done post-optimizer step to mirror the Lipschitz-transformer workflow (update → project).
  - Scaling: Using sqrt(out/in) matches the RMS→RMS induced norm scaling used in the reference, with an option to be sensitive to w_max.
  - Scope: Target nn.Linear weights in attention (wq, wk, wv, wo) and MLP (w1, w2, w3) by default. Embedding and LM head remain excluded unless you include them explicitly.
  - Performance: Methods are bf16-friendly and in-place on weight.data under no_grad. Exact SVD (svd_cap) is available but costlier.

  What’s next (optional, I can do this next)

  - Per-submodule control: per-head/per-layer w_max and method (e.g., Q,K,V different from proj/MLP).
  - Metrics: Log per-layer spectral norms and weight/grad norms for audits.
  - Muon integration: Add orthogonalized momentum and more precise coupling to match the nanogpt reference.
  - LLaMA/Qwen3 variants: Surface config presets and ready-to-run scripts for different sizes.

  If you want, I can:

  - Wire per-layer selection or per-role w_max (Q/K/V vs proj vs MLP).
  - Add spectral norm logging to metrics output.
  - Extend to LLaMA/Qwen3 arch variants and provide small unit tests for the projector.