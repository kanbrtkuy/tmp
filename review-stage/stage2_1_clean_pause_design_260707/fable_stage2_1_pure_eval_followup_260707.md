# Fable Follow-up: Stage2.1-pure Eval Gate

Date: 2026-07-07

Context:
- Added Stage2.1-pure model-comparison eval configs for 1.5B and 8B.
- Fable initially found one blocker: the 1.5B eval config used `judge_backend: vllm` but did not override WildGuard to the vLLM-compatible `wildguard_vllm_head_dim128` path.
- Fixed the 1.5B config to use the same explicit `model_map` as 8B.
- Re-ran the 1.5B dry-run and confirmed judge commands point to `/workspace/models/judges/wildguard_vllm_head_dim128`.

## Raw Review

判定基本 OK:1.5B 用与 8B 完全相同的 judge model_map(wildguard → `/workspace/models/judges/wildguard_vllm_head_dim128`,llamaguard/harmbench 显式路径)正是保证两个规模判分可比的正确做法,dry-run 里 judge 命令已指向 head_dim128 版本说明映射已生效。剩余需要确认的只有三点:(1)实跑前用 `ls`/`du` 确认 A6000 上 wildguard_vllm_head_dim128 目录完整存在(只查元数据,不读内容);(2)确认 1.5B eval 的生成端与 8B 用同一套 decoding 参数和 pause 配置(单一 `<|pause|>`×3,无解码约束),避免判分可比但生成不可比;(3)48GB 单卡上 judge 与生成模型不要同时驻留显存,按顺序加载即可。若这三点无异常,可以直接实跑。
