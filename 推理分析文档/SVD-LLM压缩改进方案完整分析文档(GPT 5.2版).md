# SAES-SVD 改进方案（基于 SAES-SVD / ERC-SVD / NSVD / MGAA / FLAT-LLM / ProcrustesGPT 的“可证明 + 易落地”设计稿）

> 目标：在 **不（或极少）训练** 的前提下，改进 SAES-SVD 的压缩质量/稳定性/可解释性，并保证改动具备“数学理论容易写清楚、工程实现容易复现”的特征。  
> 使用方式：后续你只要把这份文档丢到新对话，我就能直接按其中某个方案进入实现/写论文阶段。

---

## 0. 六篇论文里对“改 SAES-SVD”最有用的模块（浓缩版）

### 0.1 SAES-SVD：两件事
1) **CEALC（累积误差感知的层压缩）**：把“上游累计误差”显式写进当前层的压缩目标，并给出白化后截断 SVD 的闭式解（核心是构造一个随 β 变化的目标矩阵再做 Trunc-SVD）。:contentReference[oaicite:0]{index=0}

2) **ACES（自适应协同误差抑制）**：针对每层选择最优 β，使目标矩阵在给定 rank 下 **RER（Retained Energy Ratio）最大**（能量尽量集中在 top-r 奇异值里）。直接暴力扫 β 每次都要重做 SVD 太贵，于是提出 **FS-FOA** 近似 + 闭式求解 β：把 tail ratio 写成 β 的二次式比值，求导后得到一个二次方程，枚举根+边界选最优 β\*。:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

> 关键公式（你后面写论文会反复用）  
> - RER 目标（等价于最小 tail ratio）：β\* = argmax RER(β):contentReference[oaicite:4]{index=4}  
> - tail ratio 与 FS-FOA：ρ(β)、\tildeρ(β):contentReference[oaicite:5]{index=5}  
> - stationary 二次方程与选择规则 + α↔β 映射:contentReference[oaicite:6]{index=6}

---

### 0.2 MGAA：两级预算分配（很适合接在 SAES 前面当“预算模块”）
- **子层重要性**：用子层输入/输出特征的平均余弦相似度 I 衡量（低相似度→变化大→更重要→少压缩）。:contentReference[oaicite:7]{index=7}
- **把重要性映射成压缩率**：Z-score 标准化 + 平移满足全局压缩率 pt（公式 13/14），并用 15/16 进一步修正保证整体目标满足结构比例（FFN/MHA 参数量差异）。:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
- **子层内矩阵分配（Energy-balanced）**：对每个矩阵算能量累计曲线 c，要求同一子层内不同矩阵在各自 rank 下 **保留能量比例接近**，并在 rank 预算下最大化总保留能量（式 19 + 约束）。:contentReference[oaicite:10]{index=10}

---

### 0.3 ERC-SVD：两段式“残差补偿”很便宜，但经常很顶
- 把目标 rank r 分成 **ri（主低秩）+ rr（残差低秩）**：先对 W 做 Trunc-SVD 得 Wr_i，再对残差 R=W−Wr_i 再做一次 Trunc-SVD 得 Rr_r，最后把两套低秩因子拼起来。算法 2 甚至把“吸收奇异值 + 合并因子”写成固定操作流，非常好抄到工程里。:contentReference[oaicite:11]{index=11}:contentReference[oaicite:12]{index=12}
- 另一个关键：**只压最后 k 层（partial-layer compression）**，用来减轻误差传播。:contentReference[oaicite:13]{index=13}

---

### 0.4 NSVD：嵌套分解 = “对抗输入分布漂移”的训练无关增强
- NSVD 把同一层的分解做成“嵌套”：第一步用校准激活得到一个分解，第二步再把分解嵌进去，在另一个统计下再分解一次，本质上是在逼近 **不同输入分布下都好用** 的权重近似。
- 论文给出误差界（你写理论时可直接套到“分布漂移鲁棒性/上游误差导致的激活漂移”）。

---

### 0.5 FLAT-LLM：把“正交投影基”吸收到 Wv/Wo，head 级细粒度压缩
- 对每个 head 的 value 输出做 PCA，得到截断基 \tilde Q_v^h，然后把 \tilde Q 吸收到 Wv/Wo：维度直接变小，但计算图仍是一次 value 投影 + 一次 output 投影（工程形态友好）。:contentReference[oaicite:16]{index=16}

---

### 0.6 ProcrustesGPT：正交变换做“预条件”，提升可压缩性
- 在不改变模型函数（利用结构不变性/等价变换）的前提下，用正交矩阵 Q 作为“旋转”，让权重更贴近某个可压缩结构；其核心优化是一个加权的 Procrustes 目标。这个思想可以迁移成“旋转后让 RER 更高/尾能量更低”。

---

## 1. 我重新推荐的 4 个方案（按“成功率×创新度×工作量”综合排序）

> 你下一步如果要我“直接开始实现某个方案”，建议优先从 **方案 A 或 B** 开始。

---

## 方案 A（优先级 1）：**MGAA-ACES：多粒度预算分配 + SAES 的逐矩阵自适应 β\***
### A1. 核心想法
SAES 的强项是“给定 rank 时，如何选 β\* 让目标矩阵更低秩（RER 更高）”。  
MGAA 的强项是“全局预算怎么分到子层/矩阵更合理”。  
把两者拼起来就得到一个非常自然的新方法：

- **先用 MGAA 决定：每层 MHA/FFN 的压缩率、每个矩阵（Wq/Wk/Wv/Wo/Wup/Wdown/…）的 rank 预算**；
- **再对每个矩阵，在该 rank 下用 ACES 闭式求 β\***；
- 最后按 SAES 的 CEALC 闭式解做分解替换。

这能同时解决：
- “β 选得好但 rank 分配烂”；
- “rank 分配还行但 β 固定导致能量泄漏”。

### A2. 数学目标怎么写（论文可直接用）
对每个矩阵（或每层）都有 SAES 的能量集中目标：
\[
\beta^\* = \arg\max_{\beta\in[0,1]} \text{RER}_r(G(\beta))
= \arg\max_{\beta\in[0,1]}\frac{\sum_{i\le r}\sigma_i(G(\beta))^2}{\sum_i\sigma_i(G(\beta))^2}
\]
:contentReference[oaicite:19]{index=19}

MGAA 的子层/矩阵预算目标可以写成“先分配压缩率 p，再分配子层内 rank”：
- 子层重要性：I（余弦相似度）:contentReference[oaicite:20]{index=20}
- 压缩率映射：z-score + 平移满足 pt:contentReference[oaicite:21]{index=21}
- 子层内 energy-balanced：同一子层内矩阵的能量保留比例一致（式 19 约束），并在预算下最大化能量:contentReference[oaicite:22]{index=22}

**新的统一视角（你的创新点之一）**：  
把 MGAA 的能量 c_{r_a} 从“PCA 特征值能量”替换为 SAES 的 **RER_r(G_a(\beta_a^\*))**，相当于让“预算分配”直接对齐 SAES 的压缩目标，而不是对齐一个间接代理指标。  
（理论上非常好写：RER 就是奇异值能量占比，和 SAES 的目标完全一致。）

### A3. 详细实现步骤（工程落地版）
**输入**：原模型 M、校准数据 C、全局目标压缩率 pt、MGAA 超参 α_MGAA、SAES 的每矩阵候选 rank（或由 pt 推出）。  
**输出**：压缩模型 M'

**Step 0｜校准跑一次（拿到激活）**
1. 采样校准数据 C，跑 forward，缓存/流式统计所需的中间量。

**Step 1｜MGAA：子层级压缩率分配**
2. 对每个 sublayer（MHA/FFN）取输入特征 X、输出特征 Y，按式 (12) 算重要性 I（列向量余弦相似度平均）。:contentReference[oaicite:23]{index=23}
3. 用 z-score (13) 标准化，再用 (14) 得到每个 sublayer 的压缩率 p，并按 (15)(16) 做一次校正保证全局 pt 与结构比例一致。:contentReference[oaicite:24]{index=24}:contentReference[oaicite:25]{index=25}

**Step 2｜MGAA：子层内矩阵 rank 分配（energy-balanced）**
4. 对 sublayer 内每个矩阵 W_a，算其能量累计向量 c（特征值归一化 17、累计 18）。:contentReference[oaicite:26]{index=26}
5. 在子层 rank 预算 R_budget 下，求解 MGAA 的 constrained allocation（式 19）：  
   - 约束：不同矩阵的 c_{r_a} 彼此接近（|c_{r_a}-c_{r_b}|<ε）  
   - 目标：最大化 Σ c_{r_a}  
   得到每个矩阵的 rank r_a。:contentReference[oaicite:27]{index=27}

> **建议实现法（不必真做复杂优化）**：用“二分统一能量阈值 t”很快：  
> - 给定 t，令 r_a = min{r: c_r ≥ t}；  
> - 若 Σ r_a > R_budget，则降低 t；否则升高 t。  
> 这能近似满足“能量一致 + 预算约束”，也更好写进论文（可证明单调性）。

**Step 3｜SAES：逐矩阵 CEALC + ACES（β\* 闭式）**
6. 对每个矩阵 W_a（属于第 ℓ 层某子层），按 SAES 收集二阶统计：H_ℓ、Δ_ℓ（可用流式统计避免存激活）。:contentReference[oaicite:28]{index=28}
7. 构造 SAES 目标矩阵：G_ℓ(β)=S_ℓ+βD_ℓ（论文里 S、D 的定义见式 (13) 的描述）。:contentReference[oaicite:29]{index=29}
8. 用 FS-FOA 近似 tail ratio：  
   - 真实 tail ratio：ρ(β)=sqrt(E(β))/||G(β)||_F；E(β)=Σ_{i>r}σ_i^2(G(β)):contentReference[oaicite:30]{index=30}  
   - 近似：\tildeρ(β)=||S⊥+βD⊥||_F / ||S+βD||_F:contentReference[oaicite:31]{index=31}
9. 把分子分母展开成 β 的二次式，解 stationary 二次方程 (49)，再按“根+边界”枚举选最小 \tildeρ 的 β\*（再由 α↔β 做映射如果需要）。:contentReference[oaicite:32]{index=32}
10. 用该矩阵最终 rank r_a（来自 MGAA）与 β\*_a，对 G 做截断 SVD，得到 CEALC 的闭式分解替换权重。:contentReference[oaicite:33]{index=33}

### A4. 论文创新点/贡献表述（可直接摘抄）
- **贡献 1（统一框架）**：提出 MGAA-ACES，把“全局预算分配（MGAA）”与“逐矩阵能量集中（SAES-ACES）”统一到同一类 **能量占比最大化** 目标下，实现训练无关的端到端协同压缩。  
- **贡献 2（逐矩阵 β\*）**：把 SAES 的 β 自适应从“逐层”拓展到“逐矩阵/逐子层”，在相同参数预算下进一步提升 RER，减少能量泄漏到被截断子空间。  
- **贡献 3（可证明 + 可复现）**：β\* 由 FS-FOA 导出二次方程闭式求解；rank 分配用能量累计曲线单调性可二分求解，整体复杂度低、实现稳定。

### A5. 涉及论文
SAES-SVD（CEALC/ACES/FS-FOA）、MGAA（两级预算分配）。

---

## 方案 B（优先级 2）：**ERC-ACES：在 SAES 后加一段“残差补偿 rank”**
### B1. 核心想法
SAES 通过 β\* 让目标更低秩，但在高压缩比时仍可能存在明显的 **local truncation 残差**。  
ERC-SVD 的经验是：**再花很小的 rr rank 去拟合 residual，性价比极高**。  
因此我们做一个非常简单但强力的改法：

> 对每个矩阵：先用 SAES 得到主低秩近似，再对 residual 做第二次低秩补偿；最终把两套低秩因子合并。

### B2. 数学/结构怎么写
- 第 1 段：SAES 得到 \hat W_main（rank r_i）  
- 残差：R = W − \hat W_main  
- 第 2 段：R ≈ \hat W_res（rank r_r）  
- 最终：W ≈ \hat W_main + \hat W_res（总 rank 近似 r_i+r_r）

ERC-SVD 的算法 2 提供了标准流程：  
先对 W 做 SVDTRUNC 得中间近似，再算 residual，再对 residual 做 SVDTRUNC，吸收奇异值并合并因子。:contentReference[oaicite:34]{index=34}

### B3. 详细实现步骤
**输入**：模型 M、校准数据 C、每矩阵总 rank r、残差 rank 比例 η（或直接指定 rr），SAES 的统计与 β\* 求解。  
**输出**：压缩模型 M'

1) **主分解（SAES）**：对矩阵 W，按 SAES 的 CEALC+ACES 得到一对因子 (U_i, V_i)，rank=r_i。  
2) **残差计算**：R = W − U_i V_i（注意这里按你实际存储形式：若存的是 (U√Σ, √ΣV^T)，重构要一致）。  
3) **残差分解（ERC-style）**：对 R 做一次 Trunc-SVD 保留 rr，得到 (U_r, V_r)，并吸收奇异值。:contentReference[oaicite:35]{index=35}  
4) **合并**：按 ERC 的 ComB 操作把两套因子在 rank 维拼接（等价于把两项相加）。:contentReference[oaicite:36]{index=36}  
5) **可选增强（partial-layer）**：只对最后 k 层使用“SAES+残差补偿”，前面层保持 SAES 或更轻压缩，以减少误差传播（沿用 ERC 的 partial-layer 思路）。:contentReference[oaicite:37]{index=37}

### B4. 论文创新点/贡献表述
- **贡献 1（更强的 local-error 控制）**：在 SAES 的“能量集中 + 累积误差补偿”之外，引入二段式 residual 低秩补偿，显式提升局部重构精度。  
- **贡献 2（更强的可控性）**：通过 rr（或 η）提供一个简单旋钮，在同等总参数量下能更灵活地在“主结构”和“细节残差”之间分配容量。  
- **贡献 3（工程友好）**：残差补偿完全沿用 ERC-SVD 算法模板，可复用现成代码路径，风险低。

### B5. 涉及论文
SAES-SVD、ERC-SVD。

---

## 方案 C（优先级 3）：**NS-ACES：把 SAES 的“上游误差导致输入漂移”显式做成“嵌套分解鲁棒项”**
### C1. 核心想法
SAES 已经把上游误差通过 Δ 引入目标，但它仍主要在“当前统计”下求闭式解。  
NSVD 的嵌套分解提供一个直接工具：**让同一层的近似同时对两种输入分布都稳**。

最契合 SAES 的视角是：
- 分布 1：原始校准输入 X  
- 分布 2：压缩传播后的输入 X'（或用 SAES 的累计误差参考来构造）

然后做“嵌套分解”，得到对分布漂移更鲁棒的低秩替换。

### C2. 数学写法（最省事版本）
NSVD 的嵌套形式可以直接作为结构模板（两次分解嵌套）：
\[
\tilde W \approx (A_1B_1) \quad\Rightarrow\quad \tilde W \approx A_1(A_2B_2)B_1
\]
并且给出误差界（你可把“输入漂移幅度”映射到界里的项，作为理论支撑）。

### C3. 详细实现步骤
1) 先按 SAES 得到第一层低秩替换（主结构）。  
2) 用该替换跑一遍（或用统计近似）得到漂移后的输入分布 X'（可只在校准集上）。  
3) 固定外层因子 A1、B1，在“内层”再做一次分解（例如对中间映射或等价重参数）使其在 X' 下更优。  
4) β\* 的选择可以：  
   - **方案 C-1（更简单）**：外层仍按 SAES-ACES 解 β\*，内层只做 NSVD 嵌套分解；  
   - **方案 C-2（更强）**：外层/内层各自解一个 β\*（但写论文会更长）。

### C4. 论文创新点/贡献表述
- **贡献 1（把“累计误差”升级为“分布漂移鲁棒”）**：提出 NS-ACES，用嵌套分解显式适配压缩导致的输入分布变化。  
- **贡献 2（理论更顺）**：直接复用 NSVD 的误差界，把 SAES 的累计误差解释为“分布扰动”，从而得到端到端稳定性的理论支撑。  
- **贡献 3（实现仍然轻量）**：只需额外一轮校准 forward（或统计近似）+ 一次嵌套分解，无需训练。

### C5. 涉及论文
SAES-SVD、NSVD。

---

## 方案 D（优先级 4 / 风险更高但新颖）：**Procrustes-ACES：正交预条件让 RER 更高，再用 SAES 解 β\***
### D1. 核心想法
SAES 的 β\* 优化本质是“沿着 D 方向调目标矩阵，让 top-r 能量更集中”。  
但如果我们允许先做一个 **正交旋转 Q**（不改变模型函数或可通过相邻层抵消），就可能把能量分布“旋到更适合低秩截断”的坐标系里，再做 SAES 会更强。

ProcrustesGPT 给了现成的“求 Q 的加权 Procrustes 目标 + 结构不变性”的写法模板。

### D2. 一个最容易落地的具体形式（推荐写这个）
对某层的目标矩阵 G(β)，引入 Q：
- 旋转后目标：G_Q(β) = Q^T G(β)  
- 目标：最大化 RER_r(G_Q(β)) 或最小化 tail ratio ρ(G_Q(β))  
- 优化策略：先固定 β（例如用 SAES 得到的 β\* 或 β=0），求一个 Q；再用该 Q 下重新解 β\*。

Q 的求解可以用 Procrustes 的形式做“在校准激活加权下最小化重构误差”，写法与 ProcrustesGPT 的式 (3) 同构，论文叙事会非常顺。

### D3. 详细实现步骤（最小可行实现）
1) 校准集上拿到该层输入/输出统计，构造一个权重矩阵的加权版本（例如按输入协方差加权）。  
2) 解一个正交 Q（用 SVD 解 Procrustes 通常是闭式）。  
3) 将 Q 以“等价变换”方式插入网络（要么直接旋转权重参数并在相邻层补偿，要么只对分解目标做旋转但最终落回原坐标）。  
4) 在旋转坐标下运行 SAES：对 G_Q(β) 解 β\*（仍可用 FS-FOA 闭式），再做截断 SVD。  

### D4. 论文创新点/贡献表述
- **贡献 1（预条件思想）**：首次将 Procrustes 式正交预条件引入 SAES 类目标，显式提升 RER，从“坐标系选择”层面提高可压缩性。  
- **贡献 2（理论友好）**：正交变换保持范数与谱性质的基本不变性，配合 SAES 的 FS-FOA 近似，仍可给出可解析的 β\* 选择与误差界讨论。  
- **贡献 3（模块化）**：可作为独立前处理模块，失败时可直接退回方案 A/B，不影响主线。

### D5. 涉及论文
SAES-SVD、ProcrustesGPT（可选再引用 SVD-LLM 的白化/统计收集作为实现细节支撑）。

---

## 2. 一个“可选增强包”：把 FLAT-LLM 用在注意力 head 的 Wv/Wo，然后其余矩阵用方案 A/B/C/D
如果你后续希望“再加一个亮点”，FLAT-LLM 很适合当加分项：
- 对注意力 head 的 value 输出做 PCA，得到 \tilde Q_v^h；
- 用 \tilde Q_v^h\tilde Q_v^{hT} 近似投影并吸收到 Wv/Wo，直接改变维度，算子形态仍友好。:contentReference[oaicite:43]{index=43}

最保守的用法：
- 只对 Wv/Wo 用 FLAT（head-wise）；  
- 其余矩阵仍按 SAES（或 SAES+MGAA/残差）压缩；  
- 然后整体误差传播仍由 SAES 的累计误差项去抑制。

---

## 3. 最后给你一个“选型建议”（不需要你再分类/补材料）
- **想快速出结果（成功率最高）**：先做 **方案 A（MGAA-ACES）**。  
- **想把指标再往上顶、且工程风险很低**：在 A 的基础上叠 **方案 B（残差补偿）**。  
- **担心校准集不稳/迁移掉点**：考虑 **方案 C（NS-ACES）**。  
- **想搏一个很新的点（但调参/等价变换更费）**：做 **方案 D（Procrustes-ACES）**。

---

## 4. References（你后续写论文/新对话恢复上下文用）
- SAES-SVD（txt 版本 + 原 md 版本）  
  - :contentReference[oaicite:44]{index=44}  :contentReference[oaicite:45]{index=45}
- ERC-SVD（txt 版本）  
  - :contentReference[oaicite:46]{index=46}
- NSVD（txt 版本）  
  - :contentReference[oaicite:47]{index=47}
- MGAA（txt 版本）  
  - :contentReference[oaicite:48]{index=48}
- FLAT-LLM（md 分析 + txt 版本）  
  - :contentReference[oaicite:49]{index=49}  :contentReference[oaicite:50]{index=50}
- ProcrustesGPT（txt 版本）  
  - :contentReference[oaicite:51]{index=51}



### 主线 #1：在 SAES 上加一个“残差补偿分支”（Residual-Boosted SAES）

**核心想法**：SAES 得到每层低秩近似后，显式再加一个小 rank 的补偿项去拟合残差（NSVD/ERC-SVD 的共同内核）。

**为什么数学证明容易**
 对任意层权重 $W$ 和 SAES 得到的近似 $\hat W^{(1)}$，令残差 $R=W-\hat W^{(1)}$。
 你再求一个 rank-$r_2$ 的补偿 $\hat W^{(2)}$ 去最小化某个范数下的 $\|R-\hat W^{(2)}\|$，则

- 取 $\hat W^{(2)}=0$ 是可行解
- 所以最优解一定满足误差 **不增**：$\|R-\hat W^{(2)}\|\le \|R\|$
   这就是“误差单调下降”的一行证明（够你写 theorem/lemma）。

**怎么落地（最省事版本）**

- 第 1 段：完全沿用 SAES 的 CEALC/ACES，得到 $\hat W^{(1)}$（它本来就有截断 SVD 闭式解）。Saes-svd_self-adaptive_suppress…
- 第 2 段：对残差 $R$ 做一次小 rank 的截断（你可以做 weight-space，也可以做 activation-aware residual-space）。
- 最终：$\hat W=\hat W^{(1)}+\hat W^{(2)}$，推理就是两次低秩 GEMM 相加（也可以把两段因子拼接回单个 rank=$r_1+r_2$ 的因子，部署最简单）。

**你可以写成贡献点**

- “SAES 的误差抑制” + “显式残差子空间补偿” = 同压缩率下更低的端到端误差，尤其对分布外更稳（NSVD 的叙事）。

------

### 主线 #2：做一个“传播加权”的全局 rank/压缩率分配器（Propagation-weighted Allocation）

**核心想法**：不要只让 ACES 最大化某种能量集中（RER），而是把“该层误差会被后续放大多少”显式并入分配权重，然后做一个全局预算优化（MGAA/FLAT-LLM/SVD-LLM V2 都在往这个方向推）。

**一个非常好写的可证明框架**
 设第 $\ell$ 层压缩引入的输出误差（在 SAES 的加权空间里）可用
$$
\varepsilon_\ell^2(r_\ell)=\sum_{i>r_\ell}\sigma_{\ell,i}^2
$$
表示（尾能量；SAES/ASVD 这类白化 + 截断 SVD 的语境下，这种表达最顺）。

再给每层一个“传播权重” $g_\ell$（例如用后续层算子范数上界、或用校准激活估计的放大系数），得到端到端上界：
$$
\mathrm{Err}_{\mathrm{final}} \ \lesssim\ \sum_\ell g_\ell^2\,\varepsilon_\ell^2(r_\ell)
$$
这种上界推导非常标准、很好写。

于是你的全局分配就是：
$$
\min_{\{r_\ell\}} \sum_\ell g_\ell^2\sum_{i>r_\ell}\sigma_{\ell,i}^2
\quad \text{s.t.}\quad \sum_\ell \mathrm{cost}(r_\ell)\le \mathrm{Budget}
$$
由于每加 1 个 rank 的边际收益就是“下一项 $\sigma^2$”（天然递减），用 **贪心/优先队列**做分配几乎可以写成“最优/近最优”的结论（证明思路和水位法类似）。

**工程实现怎么接 SAES**

- 你本来就要做每层目标矩阵的 SVD（SAES 的 $G_\ell$ 目标矩阵截断 SVD）。Saes-svd_self-adaptive_suppress…
- 你顺手拿到奇异值谱 $\{\sigma_{\ell,i}\}$，再估一个 $g_\ell$（用少量校准 batch 就能做）。
- 然后统一决定每层/每矩阵的 rank，再回到 SAES 做一次压缩即可。

------

### 主线 #3：把“只压最后 k 层/分段压缩”纳入 SAES（结构性切断误差链）

这是 ERC-SVD 的 PLC 思想，和 SAES 的“累计误差抑制”属于互补关系：SAES 是“连续抑制”，PLC 是“结构切断”。

最省事、效果常常很强的版本：

- 固定整体压缩率 $R_o$，枚举 $k$（只压最后 k 层），用校准集测一个 probe（末层输出误差/若干层 probe），选最小的 $k$。42.ERC-SVD：Error-controlled SVD…
- 选定层集合后，对这些层用 SAES 压缩，其余层不压。
   写论文时你可以强调：**SAES 解决“怎么压得更稳”，PLC 解决“该压哪些层才不把误差传爆”**。









