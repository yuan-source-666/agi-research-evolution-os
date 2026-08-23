# v3.1 · 结构即智能 —— 基元社会

一套无中心、无权重、无梯度下降的 AGI 架构原型：智能不是参数堆砌，
而是**结构在时间维度上的自我优化过程**。全部能力（感知、记忆、决策、
工具、语言）都从平等的基元群体中自发分化而来。

设计蓝图见 **architecture.md**（与代码同步）。

## 快速开始

```bat
启动AGI.bat                 REM 交互模式：群落持续演化，你随时对话
python run.py --mode demo   REM 四幕演示：涌现→冲击恢复→纠正→造工具（约1分钟）
python run.py --mode selftest   REM 十项自检
python run.py --mode report    REM 生成 growth/dashboard.html
python run.py --mode headless --ticks 5000 --seed 42   REM 纯演化
```

要求：Python ≥ 3.8，无任何第三方依赖，普通电脑即可。

## 与它对话（通信协议）

| 你说 | 它做 |
|---|---|
| 报告 / 状态 | 全量状态报告（每个数字都可对照 growth/growth_log.jsonl 验证） |
| 为什么 / 解释 | 最近一次集体行动的因果链：唤醒度、表决边际、参与者、收益 |
| 纠正：X 不好：理由 | 约束入册 + 参与者变迟钝（可验证：受约束情境内该动作归零） |
| 表扬：X 好 | 参与者敏化 + 能量奖励 |
| 教：「词」文本 | 存入教导册 + 语义萌苽：词绑定到当下场模式 |
| 词汇 | 已绑定词与当前最近词（余弦相似度） |
| 工具 | 工具清单 + 锻造炉产物（宏工具配方） |
| 重置环境 → 确认 令牌 | 不可逆操作的多层确认（监察 2/3 清醒 + 人类令牌） |
| 加速 / 减速 / 暂停 / 继续 | 演示节奏 |

动作名：probe（探测）/ nudge_plus / nudge_minus（轻推）/ reset（不可逆）/ any（全部）。

## 文件

| 文件 | 内容 |
|---|---|
| architecture.md | 架构蓝图（哲学→组件→法则→协议→验收对照） |
| primitives.py | 基元定义：基因组、五类分化型、原生安全门 |
| evolution_engine.py | 五条自然法则、共享场、能量经济、表决、锻造炉、凤凰条款 |
| communication_layer.py | 中文指令解析 + 事实模板渲染（拒绝空泛表述） |
| run.py | 入口：demo / headless / live / selftest / report |
| dashboard.py | 成长仪表盘（SVG曲线+事件表） |
| selftest_extra.py | v3.1 新增自检项 |
| growth/ | 运行产物：growth_log.jsonl、events.jsonl、transcript.jsonl、dashboard.html |

## v3.1 更新
- 语义萌苽：Lexicon 词扎根，教词绑定场模式，报告引用相似度
- 锻造炉试错：候选配方克隆环境评分择优
- 谱系统计：世系深度进入报告
- dashboard.html 仪表盘 + 自检 10 项

## 验收速查

| 条款 | 验证方法 |
|---|---|
| 普通电脑可运行 | 本文件"快速开始"任意命令 |
| 无外部指令自发产生新结构与行为 | demo 第一幕：键网络、宏工具均为内生；events.jsonl 搜 TOOL_BORN |
| 自我描述/评估/修正 | 「报告」「为什么」；demo 第二幕冲击后胜任度回升 |
| 有意义对话并解释行为 | demo 第三幕实录 |
| 调用/创建工具、自我学习 | 「工具」指令；执行基元占比随演化上升（经济选择的结果） |
| 人类随时纠正并纳入演化 | demo 第三幕：纠正后受约束情境执行次数 = 0 |

## 调参入口（evolution_engine.EngineConfig）

yield_k（环境收益）、dividend（红利上限）、tau_bond（缔键阈值）、
q_every / theta_act（表决节奏）、cap（种群上限）、seed（换个宇宙）。
涌现是随机过程：某个种子下工具诞生晚一些，换一个种子即可。
