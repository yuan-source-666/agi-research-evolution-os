# AGI Research & Evolution OS（实验性个人学习项目）

> **[中文](README.md) | [English](README_EN.md)**

> **声明：本仓库由 AI 助手在本人指导下生成。** 代码、文档、提交均主要由 AI 完成与整理，
> 内容仅代表个人的学习探索过程，不构成成熟的研究成果，请谨慎参考。
> 仓库中所有实例 ID、访问令牌、内网地址已替换为 `REDACTED_*` 占位符。
> 完整免责声明见 [DISCLAIMER.md](DISCLAIMER.md)。

这是一个业余时间在 SCNet（国家超算互联网）海光 DCU 实例上做的自进化 LLM 学习实验。
思路是：以本地小模型（Qwen2.5-1.5B → 7B）为"大脑"，尝试用程序侧架构
（工具调用、反思循环、持久记忆、进化提案器）弥补小模型的能力不足。

## 目前做到的（与没做到的）

**做到了：**

- 一个本地中继（`relay/chat_relay.py`）→ SCNet Jupyter 内核 → 模型的对话链路
- DRAFT → CRITIQUE → REVISE → LEARN 四步反思循环与持久记忆文件
- Phase B–E 一系列实验脚本与结果 JSON（进化提案器、旋钮调参、LoRA 微调等）
- v5.x：借鉴 DeepSeek-R1 公开的思维链范式（前提锚定、自然回溯）做的程序侧数字守卫，
  在测试题上能拦截大部分数字漂移

**没做到 / 已知问题：**

- 7B 基座存在权重级的数字漂移偏见（最小化实验：无任何脚手架时把 10 记成 11），
  程序守卫只能拦输出、改不了根源
- 各 Phase 实验样本量小，结论仅为初步观察，不具备统计意义
- 代码为快速迭代的实验脚本，缺少测试与规范，多处硬编码路径

## 目录结构

| 目录 | 内容 |
|---|---|
| `relay/` | 本地中继核心（`chat_relay.py` 对话门、`rsh.py` 远程执行、监控页面） |
| `relay/ops/` | 运维脚本：守护进程部署 / 重启 / 状态检查 / 归档同步 |
| `relay/experiments/` | 实验本地侧：各 Phase 启动器、诊断探针、仿生 LLM 实验 |
| `remote/` | 云端核心：agi_core / agi_engine / evolution_loop / tool_system / memory_system / world_model，及服务脚本与状态 JSON |
| `remote/experiments/` | Phase B–E 实验云端脚本与结果 JSON |
| `remote/v8/` | 仿生 LLM v7/v8 与 DCU 消融实验 |
| `remote/agi_phase1/` | 第一阶段演化循环原型 |
| `docs/` | `RESTORE.md` 复原手册、环境信息、deepseek-harness commit 锁定 |

> 注：仓库目录为可读性做过整理。`docs/RESTORE.md` 中的路径描述的是原始工作归档，
> 与本仓库目录名可能不完全一致（原始归档中 `relay/ops/`、`relay/experiments/`
> 的脚本都平铺在 `local_scnet_v8/` 下）。

## 复原

见 `docs/RESTORE.md`。依赖：Python 3.11 / PyTorch 2.9 (ROCm/DCU)、Qwen2.5 系列模型、
`docs/deepseek_harness_commit.txt` 锁定的 deepseek-harness commit。

## License

仅供学习交流，代码按 MIT 使用（不提供任何担保），详见 [DISCLAIMER.md](DISCLAIMER.md)。
