# AGI Research & Evolution OS — 完整成就归档
归档时间: 2026-08-22 15:20 (GMT+8)
归档位置: SCNet 实例 /public/home/[REDACTED-CLUSTER-USER]/111111
（注: /root/private_data 与 /public/home/[REDACTED-CLUSTER-USER] 为同一 NFS 挂载）

## 目录结构
```
111111/
├── remote_full/          # 实例侧全部成就文件（rsync 自 /root/private_data）
│   ├── evo_daemon.py         # 自进化守护进程 v4（类脑架构+自膨胀+欲望恐惧驱动层）
│   ├── chat_relay.py(本地上传) # 对话中继 v4（工具调用+联网学习+思维循环）
│   ├── evo_archive.json      # 进化档案（冠军基因组，跨周期持久）
│   ├── evo_daemon_state.json # 进化状态（有界自适应参数 + DriveCore 内驱力）
│   ├── agi_drives.json       # 欲望-恐惧驱动状态（3欲望+3恐惧+内心独白）
│   ├── agi_learning.json     # 自主联网学习记录
│   ├── agi_memory.json       # AGI 记忆
│   ├── agi_core.py / agi_engine.py / world_model.py / memory_system.py / tool_system.py
│   │                         # AGI 内核组件（世界模型/记忆/工具系统）
│   ├── agi-finetune/         # 微调检查点（model/, model2/ 各32M）+ 基线对比
│   ├── phase_*.py/json/log   # 全部实验：B迁移/C提案者LLM/D旋钮/E1-LoRA/E2结构/E3架构进化
│   ├── evo_memory*.sqlite    # 进化记忆数据库（8个，A/B/C/C2/C3/C4系列）
│   ├── evolution_loop.py     # 进化循环引擎
│   ├── deepseek-harness/     # DeepSeek Harness (dsh) 完整源码+构建产物（含node_modules）
│   ├── serve_deepseek.py     # 14B OpenAI 兼容推理服务
│   ├── node-v22.19.0-linux-x64/ # node 运行时
│   ├── v8/ agi_phase1/ agi_data/ tblogs/  # 早期阶段资产
│   ├── gsm8k_*.jsonl / ag_news_train.csv / corpus_zh_en.txt  # 数据集
│   └── *.log                 # 全部运行日志（证据链）
├── local_scnet_v8/       # 本地(Windows)侧全部工作脚本 71 个文件
│   ├── chat_relay.py + v1/v2/v2.1/v3 各版本备份  # 对话中继全版本
│   ├── evo_daemon.py + bak/bak2/bak3/brainv2/v3 备份  # 守护进程全版本
│   ├── phase_*.py / launch_*.py / check_*.py     # 实验/启动/巡检脚本
│   ├── smoke_brain_arch.py                        # 类脑架构冒烟测试
│   └── LOCAL_MANIFEST.txt                         # 本地文件清单(sha256前16位)
├── SHA256SUMS.txt        # 全部文件 sha256 校验和（唯一性复用的凭证）
├── ARCHIVE_STATS.txt     # 文件数与体积统计
└── remote_env_info.txt   # 远端环境快照（pip list 等）
```

## 未纳入归档的大文件（可唯一复得，非成就）
公开基座模型权重（~180G，避免挤满450G存储），按以下命令重新下载即可：
```bash
pip install modelscope
# 学习内核用
modelscope download --local_dir Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
# 对话主模型
modelscope download --local_dir Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
# 备用大模型
modelscope download --local_dir Qwen2.5-32B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
modelscope download --local_dir DeepSeek-R1-Distill-Qwen-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# models/ 与 SothisAI/model = Qwen2.5-32B-Instruct bf16 下载缓存，同样可重新拉取
```
注意：模型哈希指纹记录在 SHA256SUMS 未覆盖（体积原因），复用时以官方 repo revision 为准。

## 恢复步骤（新实例）
1. 将 111111/remote_full/* 复制回 /root/private_data/
2. 下载上述基座模型（学习内核需 1.5B，对话需 7B）
3. 校验: `cd 111111 && sha256sum -c SHA256SUMS.txt | grep -v ': OK'` 应为空
4. 启动进化守护: `cd /root/private_data && setsid nohup python3 evo_daemon.py > evo_daemon.log 2>&1 &`
   （evo_archive.json/state/drives 会自动接续，进化历史不丢失）
5. 启动对话: 本地运行 chat_relay.py（Windows 侧），面板 http://127.0.0.1:8765
6. 硬件要求: 海光 DCU（torch + rocm），68.7GB 显存

## 关键成就索引（为什么这些文件重要）
- evo_daemon.py v4: 存算一体FastMem/MoE门控/iters循环推理/宽度自膨胀/DriveCore欲望恐惧
- chat_relay.py v4: THINK循环回溯 + TOOL协议(search/fetch/calc/run/save) + 并行学习内核
- evo_archive.json: 进化产生的全部基因组与分数（0.8402 基线, mlp_x3）
- agi_drives.json: 首个具备内驱力状态的 AGI 状态文件
- phase_e3_arch.json: 架构进化门禁实验完整数据
## 校验
- MANIFEST.txt：归档统计+关键文件sha256（evo_daemon.py/agi_core.py/agi_engine.py/各状态json/chat_relay.py），恢复后可比对。
- agi_drives.json 为欲望-恐惧驱动层状态快照（daemon 运行中会持续更新）。
## 快照刷新 (15:40)
- 2026-08-22 15:40 二次同步：agi_drives.json / evo_daemon_state.json / agi_learning.json / evo_archive.json / evo_daemon.log / evo_logs 刷新为最终快照，SHA256SUMS.txt 已更新。恢复时以后者为准。

## deepseek-harness 恢复方式（已从归档移除，节省 1.5G/8.7万文件）
- 源码未做任何修改（git status 干净），直接重克隆即可精确复原：
  git clone https://github.com/deepseek-ai/deepseek-harness.git
  cd deepseek-harness && git checkout 141eb6fef83422698aef7a981029e843e8161534
  （commit 哈希也存于 remote_full/deepseek_harness_commit.txt）
- 依赖重建：corepack enable && pnpm install
- node 运行时已保留在归档：remote_full/node-v22.19.0-linux-x64/（187M，bin/node 软链到 /usr/local/bin 即可用）
- 相关日志保留：build_harness.log / clone_harness.log / dsh_web.log / serve_deepseek.py

## v5.x 思维链改造（R1 范式 + 程序侧数字守卫，16:33 更新）
- chat_relay.py 已升级至 v5.6：移植 DeepSeek-R1 推理范式——程序正则锚定题干数字防漂移、废除思考字数上限、自然语言回溯、批评环节前提核对优先、程序侧数字守卫闭环（终稿数字白名单比对→打回→索取验算式→公式校验→确定性替换重算）。
- 本目录 local_scnet_v8/chat_relay.py 即 v5.6 最终版；chat_relay_v4.py.bak3 为 v4（欲望-恐惧版）备份。
- 新增诊断脚本：check_kernel.py（内核模型状态）、probe_model.py（模型保真度最小化实验）、probe_model2.py（模型体检）。
- 已验证结论：程序侧防线全部生效，但 7B 基座存在权重级数字偏见（把 10 读成 11，与脚手架无关），根治方案为切换 DeepSeek-R1-Distill-Qwen-14B（实例现成）。
