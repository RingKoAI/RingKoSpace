# RingKoSpace — v0 probe 战果与设计归档 (2026-09-06)

> 归因更正(2026-09-06晚)：A 主线 = **RNCM PURE DUAL 的 SSM 支路简化 + Mamba 式 gated linear recurrence**（conv4+SiLU+A_log/Δ+D·x），**不是 CfC**。真 CfC 本体特征(backbone(x;s) 级联 / ff1·ff2 双态插值 / 显式 Δt)均被裁掉（因非仿射不可并行，09-03 仿射定理）。'CfC-SSM 单线液态门'系误称，已弃用。真 CfC(Δt/backbone) vs 纯Δ 的增益对照 = 未验待办。

> 模型代号: RingKoSpace（工作代号 Tibo）。
> 血统: RNCM PURE DUAL(双流验证) → field/volume(矩阵态,判死) → **无残差 gated linear recurrence**（本页）。
> 只登记实验事实与已拍板项，机制逐件独立验证，禁止夹带。
>
> **术语钉（源码级归因，§8）**：本模型曾误称"单线 CfC 液态门"。逐行对照源码后确认——CfC 三独有件（state 级联 backbone / 显式 timespan Δt / ff1-ff2 双候选插值）在代码中一行不存在；A 的递推、f64 chunk scan、decay 全部继承自 RNCM **SSM 支路**，参数化改用 Mamba 配方（conv4/SiLU/A_log/Δ/D·x）。以下正文旧称"CfC 液态门/时间常数"一律按此口径读作"输入相关逐通道门（gated linear recurrence）"。

## 1. 核心方程（单线 gated linear recurrence，推理期只有一条状态流 h）

每通道一阶线性递推（Mamba 式离散化 A_log/Δ；原 RNCM SSM 支路形态），液态时间常数概念现算、**历史不 detach（窗内全梯度）**：

```
z_t    = SiLU( Wz · conv(x) )            # 内容候选（conv = 局部短记忆）
Δ_t    = softplus( Wd x + bd )           # 输入相关步长
a_t    = exp( −Δ_t · exp(A_log) )        # 每通道液态保留门（CfC 时间常数角色）
h_t    = a_t · h_{t-1} + (1 − a_t) · z_t # 单状态更新（CfC 收敛式 × SSM 递推）
y_t    = Wr·LN(h_t)·scale + dv ⊙ x        # 读：状态读出 + 当前字节直通（D·x）
```

- 无残差、无 FFN（flux/RNCM 纪律）；scale init 0.1
- 每层独立一个 h∈[B,D]；byte LM(260) tied head
- 并行: f64 分块扫描(chunk=32，指数不出界)；历史全梯度，非逐位 detach
- 状态 = 激活不是参数（记忆是学习规则的影子）

## 2. 关键修正在内（每个都是实测得来）

| 修正 | 现象 | 处理 |
|---|---|---|
| ① 初始化 | d_bias 设负（初值 −3.5）→ 液态门起始 keep≈0.97，长视界 | 初值已在代码注释 |
| ② f64 chunk scan | 长窗 exp 溢出 → NaN | 仿 rnm：f64 分块局部归一化 |
| ③ **D·x 读直通（决定性）** | 无当前 token 直通 → 死锁 4.10 平台 | 加 dv⊙x，破台直降 |

## 3. 实测数据（RTX 4060, CUDA, bf16 训练/f64 scan）

配置 dim512 / L8 / seq256 / bs16 / AdamW 4e-3 cosine → 4e-4，warmup 30，数据 skypile jsonl（EOS 分隔）

### 3.1 收敛
```
params=6,470,656   30k tok/s
eval: 5.10 →(D·x前死锁 4.10)→ 3.11@400步 → 2.83 → 2.58 → 2.31 → 2.11 → 2.00@1800步
grad: 全程 0.4–1.1 无尖刺；无 NaN
```

### 3.2 与历史基准对照（同源同数据口径）
| 基准 | 数值 | 说明 |
|---|---|---|
| byte 1-gram 熵 | 2.889 | volume **从未跌破** |
| byte 2-gram 熵 | 2.031 | 统计上限（只看前 1 字节）|
| volume dim512/g12/25k步 | 2.958 平台 | 矩阵态单步 key → 锁死 1-gram |
| **RingKoSpace v0** | **2.004 @ 740 万 token** | **跌破 2-gram 熵** → 用上 >1 字节历史 |

→ 判定：同一预算下这是 volume 到不了的表达能力区；方向成立。

### 3.3 carry 跨窗探测（窗口 256，2048-token 连续流）
```
from-zero vs carry: delta ≈ ±0.0002（≈0，无价值也无害）
```
→ 自然文本 next-byte **不给长程记忆梯度压力**，模型主动选择不跨窗记。
“状态能不能存住精确长程条目”= next-byte 问不出来的问题 → 需合成记忆任务。

## 4. 架构判定记录
- SSM 单线无上下文开销（推理 O(1) 状态，恒定 D×层数）；代价 = 摘要型遗忘，越久越糊
- 上下文开销 = 0；精确点名召回能力 = **未测**（下一步）
- 训练期窗口化(seq256)是事务不是推理开销；长上下文推理用 carry 逐窗传即 O(1)/token

## 5. 待办（验收门槛，先测再合）
1. [ ] 合成记忆训练 → recall@distance（液态门精确记忆上限）
2. [ ] Sketch 旁路读口是否补 recall 缺口（用 1 的数据决定，不预装）
3. [ ] CfC 显式 Δt / 位置注入（vs 纯输入相关 Δ）
4. [ ] EMA(JEPA) 目标训练：权重 EMA teacher + multi-horizon latent prediction（byte-CE 保小权重作防坍缩锚）
5. [ ] 长 seq（512/1024）+ 更大 dim 看能否压过 2.03 更高阶

## 6. 复现
```
cd RingKo/RingKoSpace
/storage/Projects/RingKoAI/RingKo/discard/rnm/.venv/bin/python ssm_probe.py \
  --steps 1800 --dim 512 --layers 8 --seq 256 --batch 16 --max-tokens 9000000
```
代码: RingKo/RingKoSpace/ssm_probe.py

## 7. 5090 长跑结果（2026-09-06 晚，RTX 5090 32G，corpus.bin）

### A（纯 CfC-SSM 单线）—— 完结
配置: dim512/L8/seq512/bs12/15M token(corpus.bin)/~295s，bf16
```
eval: 1.99@1250 -> 1.92@1500 -> 1.88@1750 -> 1.83@2000 -> 1.82@2250 -> 1.795@2500(未完仍降)
```
- **1.795 大幅跌破 byte 2-gram 熵 2.031**（此前 v0 seq256 到 2.004）
- carry 跨窗探测首次出现**负 delta**（有用）：
  `stream1 carry-zero=-0.0094  per-window 有 -0.017/-0.031/-0.013...`
  `stream2 = -0.0059` —— 长 seq+长训后，跨窗状态开始被真正利用（此前 ≈0）
- 里程碑: 最低 byte 语言建模记录 + 记忆开始跨窗工作

### C（CfC-SSM + EMA 慢目标）—— 修复后运行中
- 第一版 bug：KL 蒸馏到随机 EMA teacher 的启动噪声 → loss 冲 8.2/grad 90+，eval 卡 5.4
- 修复：KL 前 200 步关闭 + 权重 0→0.3 热身(600步) + teacher momentum 0.99→0.995 跟进
- 现状: 500 步 eval 3.99 正常下降，待 2500 步对比 A

### C 结论（2500 步完结）
- final eval **3.53** vs A **1.795**；carry +0.009（有害）vs A −0.009（有益）
- **当前 EMA 实现（同位置 softmax 蒸馏）对语言建模是净伤害 → 否，不进主线**
- 合成 recall 上 C 的小增益不可外推；真 JEPA（predictor 预测未来表征 h_{t+k}，multi-horizon）尚未实现，若再探 EMA 需换此形式并作独立实验
- 主线 = **纯 A（CfC-SSM 单线）**

### 68M 收官（A 主线, D=1280/L14/seq768/bs8/corpus.bin 前 300M 段）
- 16000 步 / 98M token / 42 min / RTX 5090 / ~38.6k tok/s
- eval: 5.21 -> 2.14@1k -> 1.57@5k -> 1.38@10k -> 1.347@11k -> 1.2835@15k -> **1.2782@final**
- carry 跨窗(final, window 256): stream1 delta **-0.0188** (from-zero 1.567 vs carry 1.548)
- 跨窗记忆随规模增强: dim256 ~0 -> dim512/6.5M -0.009 -> dim1280/68M **-0.019**
- 同域基线(非 memory 旧口径): 1-gram 4.065 / 2-gram条件 2.834 -> 1.278 意味着大幅压过 2-gram、吃到高阶
- 生成: 中文文体/政经词块成形（8k<10k<15k<final 逐档变真），greedy 循环/长文退化 = exposure bias，语义未成
- **结论**: 架构"窗口预测"极强；"闭眼续写"需 streaming-state 训练(下一代主线)，非架构缺陷
- 资产: ckpt/step-{1000..15999}.pt（每千步全量档，远端+本地已存部分）+ ssm_probe/recall_probe/gen_from_ckpt 工具链
