# Sketch — LSH 内容寻址索引记忆（机制自包含报告）

> 用途：交给外部 AI（GPT 等）做独立评审。本文不假设读者有本地代码上下文，所有数据结构、算法、公式、历史数值均尽量完整给出。源代码在 `RingKoAI/RingKo/discard/field/`（field 早期版本的存储核），已被判"存储型记忆"归档，本文客观复述 + 列出与 RingKoSpace 结合的候选方案，供评审者裁决。

---

## 0. 一句话

Sketch = 把 Transformer 的"全文 KV cache 检索"换成**固定大小的哈希桶索引**：写入时用 LSH 把 key 分桶、每桶保留少量代表(medoid)，查询时只在该 key 命中的桶里做 softmax 加权取回。状态不随上下文增长（O(1) 有界），访问旧记忆**不扫描全上下文**，是"无上下文开销的精确内容寻址"。

## 1. 它要解决的问题（动机）

- Attention 的记忆 = 把每个 token 的 K/V 原样缓存，查询要跟**全部历史**比相似度：成本 O(T)、显存 O(T)、且随上下文线性涨。
- RNN/SSM 的记忆 = 定长向量"摘要"，每步压缩：O(1) 但**越旧越糊、精确条目取不回**（指数遗忘）。
- Sketch 想提供第三种：**定长、可流式、且能"点名取回"特定旧条目**的记忆（像外接 RAG 向量库，但内嵌在可训练网络里）。

## 2. 数据结构（完全定义）

参数常量：
```
num_tables      = 4       # LSH 哈希函数个数（多表降碰撞）
bucket_bits     = 8       # 每表 2^8 = 256 个桶
slots_per_bucket= 4       # 每桶最多 4 个"代表"
temperature     = 0.1     # 桶内 softmax 温度（读权重）
```

每批次状态形状（SketchState，全部 buffer/张量）：
```
key_slots   [B, num_tables, num_buckets, slots_per_bucket, D]   # 存过的 key（L2 归一）
value_slots [B, num_tables, num_buckets, slots_per_bucket, D]   # 对应的 value（L2 归一）
centers     [B, num_tables, num_buckets, D]                     # 每桶 key 的累加和（滑动中心）
counts      [B, num_tables, num_buckets]                        # 每桶写入次数
```
静态（非状态，可学习参数只在这些投影里）：
```
hash        [num_tables, D, bucket_bits]   # 固定随机高斯投影（随机初始化后冻结，不可学习）
bucket_w    = 2^[0..bucket_bits)            # 符号位 → 桶号的二进制权重
readout     Linear(D→D)，eye 初始化          # 读出层
```
模型外另有（SketchCore 投影包装）：query/key/value/skip 四个 Linear(D→D)。

## 3. 算法

### 3.1 哈希（内容寻址的核心）
```
bits(x)     = (x·hash > 0) ∈ {0,1}^bucket_bits          # 符号投影，逐表
bucket(x)   = Σ bits(x) * bucket_w                       # 得桶号 ∈ [0,256)
```
性质：语义相近（高 cos 相似）的 key 以较高概率落入同桶（LSH 保相似性）；多表 + 每桶多槽控碰撞。

### 3.2 写入（medoid 淘汰，不无界增长）
新 (key,value) 到桶后：
```
dist(slot)    = ‖key_slot − center‖₂            # center = centers/counts（滑动均值）
far           = argmax_slot dist(...)            # 当前最"偏"的代表
if counts ≥ slots 且 ‖key − center‖ < ‖far_slot − center‖:
    写进 far 槽（替换最不具代表性的）
else:
    写进第一个空槽（counts 未满）
centers += key ; counts += 1                     # 中心增量更新（不是重算）
```
所有 key/value 写前 L2 归一（相似度即 cos）。

### 3.3 查询（桶内 softmax，只碰命中桶）
```
q = L2(query)
只取该 q 命中的 num_tables×slots_per_bucket 个候选（不是全历史！）
sim(q, slot_k) = q·slot_k.key
occ           = 槽是否被占用
w             = softmax(sim / temperature)  （空槽 −∞；整桶全空 → 输出 0）
out           = readout( Σ_k w_k · slot_k.value )
```

### 3.4 复杂度
- 写入/查询都只碰**一个桶**：O(tables × slots × D)，常数，与已见 token 总数无关。
- 推理状态固定：`B × 4 × 256 × 4 × D` 一组有界槽位，可流式增量，不随上下文长。

## 4. 梯度与训练（难点在这）

- 桶分配是 **hard（离散）**：`bucket()=符号投影取整`，不可微。训练中桶分配作为"路由决策"被当作直通（hash 固定，query 的桶由其自己投影决定，梯度不通过桶号选择进入 hash；桶号对查询是确定函数）。
- 写入是 in-place 状态更新（medoid 替换），PyTorch 正常 autograd 不友好，因此对 Triton 融合路径写了**手工 autograd**：
  - `WriteStepFn`：前向 in-place 写；反向只把梯度回传"被写入/被替换槽"对应的 key/value/center（medoid argmax、one-hot 选择当 pass-through，不做该离散分支的导数）。
  - `QueryStepFn`：反向重算桶内 softmax 权重，把读出梯度拆回 query 与命中的 key/value 槽。
- 即：**读取路径可微、写入的"替换哪个槽"离散不可微（pass-through）**，可学习的只有读 + query/key/value 投影 + readout，槽本身不是参数。

## 5. 工程
- `triton_ops.py`：`_write_step_kernel`（一个线程块处理一个 (batch,table)，静态展开 S 槽）、`_query_step_kernel`（逐表逐槽加载 → 数值稳定 softmax），推理走 in-place，训练走 autograd Function。
- 端到端/推理吞吐提升显著（历史记录 5.9x / 与 dense 差距 66x→2.7x）。

## 6. 历史时间线与数值（field 早期）

```
9416330  LSH sketch index memory（内容寻址存储，替换纯线性注意混合）
bd091ca  向量化 write/query
1765acb  SketchCore 作为序列混合核心接入（FieldModel）
66ebed8  CfC timespan 连续时间位置编码（同仓并行流）
e2fa065/24baeb1/47823aa  Triton 融合 write/query（含 autograd 对）
6e63335  数值修复：写入前 value 也 L2 归一（此前不稳）
```
实测/记录要点（来自分析文档，同源）：
- 召回 0.94+，小规模训练稳定（loss 22.6→3.1 口径来自 field 报告，dim 约 320）
- 关键修复：value 未归一 → 写路径数值不稳（6e63335 一行修复）
- 已知瓶颈：medoid 淘汰逐步依赖 → 序列级并行难；桶内小 softmax 曾占 ~85% 计算（后融合）
- 方法论结论：当时"tiny 规模无质量优势"的判词成立，但被明确标注为**未覆盖其真正卖点**

## 7. 为什么被归档"存储型记忆"（客观复述）
- 09-05 方向裁决：新模型方向"记忆 = 学习 = 指数更新本身，不设仓库" → 把 Sketch（明确有仓库/槽位/召回）与 D×D 矩阵态一起判为"存储型记忆"丢进 discard。
- 但报告同时白纸黑字记着：**"真实卖点 = O(1) 推理状态 + 有界显存流式（长上下文 needle recall 主场）—— 从未测过"**（`analysis/20260904-index-table.md`）。所以它是在卖点未验证的情况下被弃，不是因为证明失败。

## 8. 与三类记忆的对照（评审用）

| | Attention KV | SSM 液态状态 | KDA 矩阵 | **Sketch** |
|---|---|---|---|---|
| 状态规模 | O(T) | O(1) 摘要 | O(D²) | **O(1) 有界槽位** |
| 精确取回旧条目 | ✅全 | ❌ 衰减糊 | 部分(线性读) | ✅ 命中即取 |
| 查询成本 | O(T) | O(1) | O(D²) | **O(桶)** |
| 是否可微 | ✅ | ✅ | ✅ | 读✅ 写路由 pass-through |
| 可流式增量 | 否 | ✅ | ✅ | ✅ |
| 每 token 写入 | 无 | 压缩 | delta 修正 | 桶替换 |

## 9. 与 RingKoSpace（单线液态 SSM）结合的候选方案（请评审）

背景：RingKoSpace v0 = 单状态 SSM（CfC 液态门 a_t 折入），无残差，byte-CE 已训到 2.004（跌破 byte 2-gram 熵 2.031，超越 volume 平台 2.958），推理 O(1)。已确认：自然文本 next-byte **不给长程记忆压力**，carry≈0。尚未测"精确长程召回"。

候选接入（都避免第二状态流 / 大杂烩）：

- **方案 X（旁路读口）**：主状态仍 SSM。在每层读口加 Sketch 增强：
  `out = gate1·SSM读(h) + gate2·readout(Sketch.query(q))`，q 由当前隐态生成。Sketch 存的是"该层观察到的 (key,value) 轨迹"，写入用当层 key/value（**写/读都只是当前输入的函数，不进 SSM 状态方程**）。作用 = 给"精确点名取回"一个**可训练但旁路**的通道，仍一条主状态。
- **方案 Y（写入增强）**：Sketch 不直接读出，而是把它命中取回的 value 作为 SSM 写候选 `z_t` 的一个加性项：`z_t = SiLU(Wz·conv(x)) + β·Sketch.query(q)`——即检索结果影响"往状态里收敛什么"。
- **方案 Z（不接，只用合成记忆任务先量化液态门极限）**：如果 recall@距离显示液态门本身够用（在给足记忆压力后），Sketch 就不需要。

## 10. 交评审的问题清单
1. Sketch 的离散桶路由在端到端训练里是否是致命伤？有没有已知近似（STE/软路由）值得试？
2. 有界槽位 + medoid 淘汰在长流式下会丢哪些信息？"无上下文开销"这个卖点在语言建模上是否真实存在价值（相对 SSM 液态 + 长窗训练）？
3. 方案 X vs Y vs Z 更推荐哪个？门控系数要不要按层/按头学？
4. 如果给足"记忆压力"（合成 recall 训练），纯 SSM 液态门能到多远的精确召回？有没有业界近似经验（DeltaNet/KDA/retrieval-augmented SSM，如 Griffin/RAG 内化）可参照？
5. EMA/JEPA 目标 + Sketch 旁路读口是否互斥？检索增强会不会破坏"单向量"表征自洽？
