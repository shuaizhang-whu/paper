我先按你现有“验算过程”的结构来重写，但把前面已经确认的问题一并修正进去：Stage 2 统一成“先 ridge 补偿，再做 $H_\ell$-度量低秩逼近”，同时把 $\beta$ 选择里的护栏补进伪代码。

下面给你一版**修正后的完整算法验算过程**，写法尽量模仿《9）完整算法验算过程.md》的结构，但把前面已经确认的问题都统一修正了：
 1）Stage 1 的 Fisher 部分改成“经验 Fisher / KFAC 因子估计”，不再写成对单个梯度直接做 Kronecker 分解；
 2）$\beta$ 选择补上候选根、区间裁剪和护栏；
 3）Stage 2 统一成“先求 ridge 补偿 $\Delta W_\ell^{LS}$，再做 $H_\ell$-度量下的 rank-$r_2$ 低秩逼近”，不再把输出残差 $E_\ell$ 直接拿去白化。这样和你当前确认过的 SAES / GFWSVD / Stage 2 修正版逻辑是一致的。

# 1）修正后的完整算法验算过程

------

## 完整算法流程：FR-SVD（Fisher-Residual SVD）

------

### Phase 0：数据准备

**输入**：

- 原始模型权重 $\{W_\ell\}_{\ell=1}^L$
- 校准数据集 $D$
- 每层 rank 分配：$r=r_1+r_2$

其中：

- $W_\ell \in \mathbb{R}^{m\times n}$
- $X_\ell, X_\ell^f \in \mathbb{R}^{n\times p}$
- $p$ 为当前层校准样本数（或 token 数）

------

**Step 0.1**：用原始模型在校准集 $D$ 上前向传播，逐层记录全精度激活
$$
\{X_\ell^f\}_{\ell=1}^L
$$

------

**Step 0.2**：初始化压缩路径激活
$$
X_1 \leftarrow X_1^f
$$
因为第一层之前没有累积误差，所以初始压缩输入与全精度输入一致。

------

### Phase 1：逐层压缩（for $\ell = 1$ to $L$）

------

## Stage 1：Fisher-CEALC 主压缩

------

### Step 1.1：构造二阶统计量

定义阻尼输入协方差与累积误差协方差：
$$
H_\ell = X_\ell X_\ell^T + \lambda_H I \in \mathbb{R}^{n\times n}
$$
**维度验算**：

- $X_\ell \in \mathbb{R}^{n\times p}$
- $X_\ell^f \in \mathbb{R}^{n\times p}$
- $X_\ell X_\ell^T \in \mathbb{R}^{n\times n}$ ✓
- $(X_\ell^f - X_\ell)X_\ell^T \in \mathbb{R}^{n\times n}$ ✓

**特殊情况**：

第一层 $X_1 = X_1^f$，因此
$$
\Delta_1 = 0
$$
故第一层退化为纯 Fisher-whitened CEALC 压缩。

------

### Step 1.2：估计经验 Fisher 的 Kronecker 因子

对当前层 $W_\ell$，定义 batch 梯度：
$$
G_i = \nabla_{W_\ell} L(\text{batch}_i) \in \mathbb{R}^{m\times n}
$$
向量化后：
$$
g_i = \operatorname{vec}(G_i)\in\mathbb{R}^{mn}
$$
经验 Fisher 定义为：
$$
I_F^{(\ell)} = \frac{1}{|D|}\sum_{i=1}^{|D|} g_i g_i^T \in \mathbb{R}^{mn\times mn}
$$
采用 Kronecker 近似：
$$
I_F^{(\ell)} \approx A_\ell \otimes B_\ell
$$
其中
$$
A_\ell \in \mathbb{R}^{n\times n},\qquad B_\ell \in \mathbb{R}^{m\times m}
$$
**维度验算**：
$$
A_\ell \otimes B_\ell \in \mathbb{R}^{mn\times mn}
$$
与 $I_F^{(\ell)}$ 一致 ✓

**说明**：

论文理论上可以写经验 Fisher，再写 Kronecker 近似；实际实现时不必显式构造完整 $I_F^{(\ell)}$，而是通过 KFAC / 隐式算子估计 $(A_\ell,B_\ell)$。

------

### Step 1.3：Cholesky 分解

对 Kronecker 因子加阻尼后做 Cholesky：
$$
A_\ell + \lambda_A I = L_A^{(\ell)}L_A^{(\ell)T}
$$
其中：

- $L_A^{(\ell)} \in \mathbb{R}^{n\times n}$
- $L_B^{(\ell)} \in \mathbb{R}^{m\times m}$

------

### Step 1.4：构造 CEALC 目标矩阵

根据 SAES-SVD 的 CEALC 形式，定义：
$$
G_\ell(\beta)
=
W_\ell\,(H_\ell+\beta\Delta_\ell)\,H_\ell^{-1/2}
\in \mathbb{R}^{m\times n}
$$
将其拆成基底项与扰动项：
$$
S_\ell = W_\ell H_\ell^{1/2} \in \mathbb{R}^{m\times n}
$$
因此
$$
G_\ell(\beta)=S_\ell+\beta D_\ell
$$
**维度验算**：

- $W_\ell H_\ell^{1/2}\in\mathbb{R}^{m\times n}$ ✓
- $W_\ell \Delta_\ell H_\ell^{-1/2}\in\mathbb{R}^{m\times n}$ ✓

------

### Step 1.5：变换到 Fisher 度量空间

定义 Fisher 空间中的基底项与扰动项：
$$
S_\ell^F = L_B^{(\ell)T} S_\ell L_A^{(\ell)} \in \mathbb{R}^{m\times n}
$$
于是：
$$
G_\ell^F(\beta)
=
L_B^{(\ell)T} G_\ell(\beta) L_A^{(\ell)}
=
S_\ell^F+\beta D_\ell^F
$$
**维度验算**：

- $L_B^T \in \mathbb{R}^{m\times m}$
- $S_\ell,D_\ell \in \mathbb{R}^{m\times n}$
- $L_A \in \mathbb{R}^{n\times n}$

因此 $G_\ell^F(\beta)\in\mathbb{R}^{m\times n}$ ✓

------

### Step 1.6：在 Fisher 空间中求 $\beta^*$

对 $S_\ell^F$ 做 rank-$r_1$ 截断相关的 ACES 近似分析。

先取 $S_\ell^F$ 的 top-$r_1$ 奇异向量，构造正交补投影：
$$
P_L = I_m - U_{r_1}U_{r_1}^T,\qquad
P_R = I_n - V_{r_1}V_{r_1}^T
$$
定义正交补分量：
$$
S_\ell^{F\perp}=P_L S_\ell^F P_R
$$
再定义 6 个标量：
$$
a=\|S_\ell^{F\perp}\|_F^2,\qquad
b=\langle S_\ell^{F\perp},D_\ell^{F\perp}\rangle,\qquad
c=\|D_\ell^{F\perp}\|_F^2
$$
其中
$$
\langle M,N\rangle = \operatorname{tr}(M^TN)
$$
近似误差比写为：
$$
\tilde\rho(\beta)=\frac{a+2b\beta+c\beta^2}{A+2B\beta+C\beta^2}
$$
令导数为零，得到二次方程：
$$
(cB-bC)\beta^2 + (cA-aC)\beta + (bA-aB)=0
$$
从实根中选取可行候选集，再结合边界点，得到：
$$
\mathcal{C}_\beta
=
\{\beta:\beta \text{ 是上式实根且 } \beta\in[\beta_{\min},\beta_{\max}]\}
\cup
\{\beta_{\min},\beta_{\max}\}
$$
然后取：
$$
\beta_{\text{best}}=\arg\min_{\beta\in\mathcal{C}_\beta}\tilde\rho(\beta)
$$
若实现中采用护栏，则最终使用：
$$
\beta_\ell^*
=
\rho_{\text{shrink}}\cdot \min(\beta_{\text{best}},\beta_{\text{cap}})
$$
其中：

- $\beta_{\min},\beta_{\max}$ 来自预设 $\alpha$ 区间映射
- $\rho_{\text{shrink}}\in(0,1]$ 为收缩因子
- $\beta_{\text{cap}}$ 为上限保护

------

### Step 1.7：在 Fisher 空间中做 rank-$r_1$ 截断 SVD

构造最优目标矩阵：
$$
G_\ell^F(\beta_\ell^*) = S_\ell^F+\beta_\ell^* D_\ell^F
$$
做 rank-$r_1$ 截断：
$$
[G_\ell^F(\beta_\ell^*)]_{r_1}
=
\hat U_\ell \hat \Sigma_\ell \hat V_\ell^T
$$
其中：

- $\hat U_\ell \in \mathbb{R}^{m\times r_1}$
- $\hat \Sigma_\ell \in \mathbb{R}^{r_1\times r_1}$
- $\hat V_\ell \in \mathbb{R}^{n\times r_1}$

------

### Step 1.8：映回原空间，得到 Stage 1 压缩权重

先定义 Fisher 目标变量的最优解：
$$
T_\ell^*
=
L_B^{(\ell)-T}\hat U_\ell \hat\Sigma_\ell \hat V_\ell^T L_A^{(\ell)-1}
$$
再映回权重空间：
$$
W_\ell^{(1)}
=
T_\ell^* H_\ell^{-1/2}
=
L_B^{(\ell)-T}\hat U_\ell \hat\Sigma_\ell \hat V_\ell^T
L_A^{(\ell)-1}H_\ell^{-1/2}
$$
于是
$$
\operatorname{rank}(W_\ell^{(1)})\le r_1
$$
若写成低秩因子，则可定义：
$$
B_\ell^{(1)}
=
L_B^{(\ell)-T}\hat U_\ell \hat\Sigma_\ell^{1/2}
\in\mathbb{R}^{m\times r_1}
$$
满足
$$
W_\ell^{(1)}=B_\ell^{(1)}A_\ell^{(1)}
$$

------

## Stage 2：特征空间残差补偿

------

### Step 2.1：定义全局输出对齐误差

定义 teacher/student 的全局对齐误差：
$$
E_\ell
=
W_\ell X_\ell^f - W_\ell^{(1)}X_\ell
\in \mathbb{R}^{m\times p}
$$
**维度验算**：

- $W_\ell X_\ell^f \in \mathbb{R}^{m\times p}$
- $W_\ell^{(1)}X_\ell \in \mathbb{R}^{m\times p}$

故 $E_\ell\in\mathbb{R}^{m\times p}$ ✓

------

### Step 2.2：求无秩约束的 ridge 补偿解

考虑岭回归目标：
$$
\min_{\Delta W}
\|\Delta W X_\ell - E_\ell\|_F^2 + \lambda_H \|\Delta W\|_F^2
$$
其闭式解为：
$$
\Delta W_\ell^{LS}
=
E_\ell X_\ell^T (X_\ell X_\ell^T + \lambda_H I)^{-1}
=
E_\ell X_\ell^T H_\ell^{-1}
\in \mathbb{R}^{m\times n}
$$
**维度验算**：

- $E_\ell X_\ell^T \in \mathbb{R}^{m\times n}$
- 右乘 $H_\ell^{-1}\in\mathbb{R}^{n\times n}$

因此 $\Delta W_\ell^{LS}\in\mathbb{R}^{m\times n}$ ✓

**注意**：

这里的 $\Delta W_\ell^{LS}$ 是**满秩 ridge 补偿解**，不是最终部署的低秩补偿权重。

------

### Step 2.3：在 $H_\ell$-诱导度量下做 rank-$r_2$ 低秩逼近

Stage 2 的目标统一写成：
$$
\min_{\operatorname{rank}(C)\le r_2}
\|(\Delta W_\ell^{LS}-C)H_\ell^{1/2}\|_F^2
$$
这表示：在 $H_\ell$-诱导的特征度量下，寻找 $\Delta W_\ell^{LS}$ 的最佳 rank-$r_2$ 近似。

对 $H_\ell$ 做特征分解：
$$
H_\ell = Q_\ell \Lambda_\ell Q_\ell^T
$$
其中：

- $Q_\ell \in \mathbb{R}^{n\times n}$ 为正交矩阵
- $\Lambda_\ell \in \mathbb{R}^{n\times n}$ 为正对角矩阵

定义白化后的补偿矩阵：
$$
\Delta W_{\ell,\text{white}}
=
\Delta W_\ell^{LS}Q_\ell \Lambda_\ell^{1/2}
\in \mathbb{R}^{m\times n}
$$
令
$$
C_{\text{white}} = C Q_\ell \Lambda_\ell^{1/2}
$$
则原问题等价为：
$$
\min_{\operatorname{rank}(C)\le r_2}
\|\Delta W_{\ell,\text{white}} - C_{\text{white}}\|_F^2
$$
由 Eckart–Young–Mirsky 定理，最优解为对白化矩阵做 rank-$r_2$ 截断。

------

### Step 2.4：对白化补偿矩阵做 rank-$r_2$ 截断 SVD

$$
[\Delta W_{\ell,\text{white}}]_{r_2}
=
\tilde U_\ell \tilde \Sigma_\ell \tilde V_\ell^T
$$

其中：

- $\tilde U_\ell \in \mathbb{R}^{m\times r_2}$
- $\tilde \Sigma_\ell \in \mathbb{R}^{r_2\times r_2}$
- $\tilde V_\ell \in \mathbb{R}^{n\times r_2}$

------

### Step 2.5：映回原空间，得到低秩补偿权重

将白化空间最优解映回原空间：
$$
W_\ell^{comp}
=
\tilde U_\ell \tilde \Sigma_\ell \tilde V_\ell^T
\Lambda_\ell^{-1/2} Q_\ell^T
\in \mathbb{R}^{m\times n}
$$
于是
$$
\operatorname{rank}(W_\ell^{comp})\le r_2
$$
若写成低秩因子，可定义：
$$
B_\ell^{(2)}=\tilde U_\ell \tilde\Sigma_\ell^{1/2}
\in \mathbb{R}^{m\times r_2}
$$
满足：
$$
W_\ell^{comp}=B_\ell^{(2)}A_\ell^{(2)}
$$
**这里要特别注意**：

这一步对应的是“$\Delta W_\ell^{LS}$ 的 $H_\ell$-度量下最优 rank-$r_2$ 逼近”，
 **不是**“直接对 $E_\ell$ 做 rank-$r_2$ 约束输出拟合的严格闭式解”。

------

## 组合与传播

------

### Step 3.1：组合最终权重

数学上可写为：
$$
W_\ell' = W_\ell^{(1)} + W_\ell^{comp}
$$
因此：
$$
\operatorname{rank}(W_\ell') \le r_1 + r_2 = r
$$

------

### Step 3.2：更新下一层输入激活

将当前层替换为压缩层后，运行该层前向传播，得到压缩路径的新激活：
$$
X_{\ell+1}
\leftarrow
\operatorname{RunCompressedLayer}(\text{layer}_\ell', X_\ell)
$$
**说明**：

这里更准确的实现语义是“替换当前层对应线性模块后，运行压缩后的整层 / 整个 block 前向”，而不是单纯写成一个矩阵乘法。

------

### Step 3.3：部署形式

数学上：
$$
W_\ell' = W_\ell^{(1)} + W_\ell^{comp}
$$
实现上建议保留为两个低秩分支：
$$
W_\ell' x
=
B_\ell^{(1)}A_\ell^{(1)}x
+
B_\ell^{(2)}A_\ell^{(2)}x
$$
这样不会显式物化成稠密矩阵，更符合压缩部署形式。

------

## 最终输出

经过逐层压缩后，得到最终压缩模型：
$$
\{W_\ell'\}_{\ell=1}^L
$$

------

# 2）修正后的完整 step 算法流程

下面是一版可以直接放进论文 / 方法章节的 step 流程。它和上面的验算过程保持一致。9）完整算法验算过程 2）创新点确认

## FR-SVD 完整算法流程（修正版）

```
输入：
    原始模型权重 {W_ℓ}_{ℓ=1}^L
    校准集 D
    每层 rank 分配 r1, r2（总秩 r = r1 + r2）
    超参数 λ_H, λ_A, λ_B
    β 选择护栏参数：β_min, β_max, ρ_shrink, β_cap

输出：
    压缩模型 {W_ℓ'}_{ℓ=1}^L

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 0：数据准备
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  {X_ℓ^f}_{ℓ=1}^L ← ForwardOriginalModel(D)
2:  X_1 ← X_1^f

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1：逐层压缩（for ℓ = 1 to L）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3:  for ℓ = 1 to L do

      // ===== Stage 1：Fisher-CEALC 主压缩 =====

4:      H_ℓ ← X_ℓ X_ℓ^T + λ_H I
5:      Δ_ℓ ← (X_ℓ^f - X_ℓ) X_ℓ^T

6:      (A_ℓ, B_ℓ) ← EstimateKFACFactors(W_ℓ, D)
        // 由经验 Fisher / KFAC 统计估计 Kronecker 因子

7:      L_A ← Cholesky(A_ℓ + λ_A I)
8:      L_B ← Cholesky(B_ℓ + λ_B I)

9:      S_ℓ^F ← L_B^T W_ℓ H_ℓ^{1/2} L_A
10:     D_ℓ^F ← L_B^T W_ℓ Δ_ℓ H_ℓ^{-1/2} L_A

11:     根据 S_ℓ^F, D_ℓ^F 计算 a,b,c,A,B,C
12:     求解二次方程：
            (cB - bC)β^2 + (cA - aC)β + (bA - aB) = 0
13:     构造候选集 C_β ← {区间内实根} ∪ {β_min, β_max}
14:     β_best ← argmin_{β ∈ C_β} ρ̃(β)
15:     β_ℓ^* ← ρ_shrink · min(β_best, β_cap)

16:     G_ℓ^F ← S_ℓ^F + β_ℓ^* D_ℓ^F
17:     Û, Ŝ, V̂^T ← TruncatedSVD(G_ℓ^F, rank=r1)

18:     W_ℓ^{(1)} ← L_B^{-T} Û Ŝ V̂^T L_A^{-1} H_ℓ^{-1/2}
        // 等价地，也可保存低秩因子：
        // B_ℓ^{(1)} = L_B^{-T} Û Ŝ^{1/2}
        // A_ℓ^{(1)} = Ŝ^{1/2} V̂^T L_A^{-1} H_ℓ^{-1/2}

      // ===== Stage 2：特征空间残差补偿 =====

19:     E_ℓ ← W_ℓ X_ℓ^f - W_ℓ^{(1)} X_ℓ

20:     ΔW_ℓ^{LS} ← E_ℓ X_ℓ^T H_ℓ^{-1}
        // 无秩约束 ridge 补偿解

21:     Q_ℓ, Λ_ℓ ← EigenDecompose(H_ℓ)

22:     ΔW_{ℓ,white} ← ΔW_ℓ^{LS} Q_ℓ Λ_ℓ^{1/2}
        // 在 H_ℓ-诱导度量下进行低秩逼近

23:     Ũ, Σ̃, Ṽ^T ← TruncatedSVD(ΔW_{ℓ,white}, rank=r2)

24:     W_ℓ^{comp} ← Ũ Σ̃ Ṽ^T Λ_ℓ^{-1/2} Q_ℓ^T
        // 等价地，也可保存低秩因子：
        // B_ℓ^{(2)} = Ũ Σ̃^{1/2}
        // A_ℓ^{(2)} = Σ̃^{1/2} Ṽ^T Λ_ℓ^{-1/2} Q_ℓ^T

      // ===== 组合 + 传播 =====

25:     数学上：W_ℓ' ← W_ℓ^{(1)} + W_ℓ^{comp}

26:     实现上：保留两个低秩分支
            W_ℓ' x = B_ℓ^{(1)} A_ℓ^{(1)} x + B_ℓ^{(2)} A_ℓ^{(2)} x

27:     X_{ℓ+1} ← RunCompressedLayer(layer_ℓ', X_ℓ)

28: end for

29: return {W_ℓ'}_{ℓ=1}^L
```

------

## 3）一句话提醒：这一版和旧版相比，核心修正在哪里

```
旧版 Stage 2：
    直接对白化 E_ℓ 做 SVD
    E_white = E_ℓ Q Λ^{1/2}
    （这一写法一般维度不对，也混淆了输出残差与权重补偿）

修正版 Stage 2：
    先用 E_ℓ 回归出满秩权重补偿 ΔW_ℓ^{LS}
    再对白化后的 ΔW_ℓ^{LS} 做 rank-r2 截断
    ΔW_{white} = ΔW_ℓ^{LS} Q Λ^{1/2}
```

