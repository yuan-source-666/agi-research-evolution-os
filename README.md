# AGI Research & Evolution OS

在 SCNet（国家超算互联网）海光 DCU 上从零构建的自进化 AGI 研究系统。
以本地小模型（Qwen2.5-1.5B → 7B）为"大脑"，通过程序侧架构（而非单纯堆参数）驱动其
**工具调用、自主联网学习、反思进化、欲望/恐惧驱动、R1 式长思维链**等能力。

## 系统架构

```
浏览器 → 本地中继(:8765, chat_relay.py) → SCNet Jupyter 常驻内核 → AGI 大脑
```

每条消息走 4 步自我反思循环（reflexion）：

1. **DRAFT** — 起草回复（首步逐字引用题干数字，防止前提漂移）
2. **CRITIQUE** — 自我批评：先核对前提，再验算算术（强制工具计算）
3. **REVISE** — 结合批评输出终稿（程序侧数字白名单守卫，白名单外数字打回重写）
4. **LEARN** — 提炼一条自我改进笔记，写入持久记忆（跨会话累积）

## 目录结构

| 目录 | 内容 |
|---|---|
| `relay/` | 本地中继与远程运维脚本（chat_relay.py 为对话门主体，v5.6） |
| `remote/` | 云端核心：agi_core / agi_engine / evolution_loop / tool_system / memory_system / world_model |
| `remote/` (根) | Phase B–E 系列实验脚本与结果 JSON（B 火种→C LLM提案器→D 旋钮→E LoRA微调/结构/架构） |
| `remote/v8/` | 仿生 LLM v7/v8 与 DCU 消融/扩展实验 |
| `remote/agi_phase1/` | 第一阶段演化循环原型 |
| `docs/` | 环境信息、deepseek-harness commit 锁定 |
| `RESTORE.md` | 完整复原手册（从零重建整个系统） |

## 关键实验结论

- **Phase C**：LLM 提案器自动生成进化方向，替代随机变异
- **Phase D**：进化旋钮（种群/温度/记忆窗口）实机调优
- **Phase E1–E3**：LoRA 微调 → 结构改造 → 架构自进化三连
- **v5.x 思维链改造**：移植 DeepSeek-R1 范式（前提锚定、无字数上限长思考、自然语言回溯），
  程序侧数字守卫闭环全链路生效；同时用最小化实验证明 7B 基座存在权重级数字漂移偏见
  （10 → 11），架构防线拦得住输出、拦不住根源 → 根治需换更大基座

## 复原

见 `RESTORE.md`。依赖：

- Python 3.11 / PyTorch 2.9 (ROCm/DCU)
- 模型：Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct
- deepseek-harness：`docs/deepseek_harness_commit.txt` 锁定的 commit

## 声明

仓库已全面脱敏：所有实例 ID、访问令牌、内网地址均已替换为 `REDACTED_*` 占位符。
