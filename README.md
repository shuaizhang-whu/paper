# SAES-SVD 论文仓库

本仓库包含论文 **SAES-SVD: Self-Adaptive Suppression of Accumulated and Local Errors for SVD-Based LLM Compression** 的全部相关文件，包括论文正文、基线方法分析、理论推导以及代码实现。

---

## 📂 仓库结构

```
paper/
├── markdown格式论文/               # 论文主文件及参考文献原文（Markdown 格式）
│   ├── SAES-SVD/                   # 本文论文稿（主文件）
│   │   ├── Saes-svd_self-adaptive_...compression.md   # 论文正文（Markdown）
│   │   ├── Saes-svd_self-adaptive_...compression.txt  # 论文正文（纯文本）
│   │   ├── SAES-SVD.py             # 本文算法代码实现
│   │   └── saes-svd审稿意见.txt    # ICLR 2026 四位审稿人意见（中译）
│   ├── SVD-LLM-v1/                 # 基线：SVD-LLM v1（白化截断）
│   │   ├── (无参考文献)Wang 等 - 2025 - SVD-LLM...md
│   │   └── SVDLLMv1.py
│   ├── SVD-LLM-v2/                 # 基线：SVD-LLM v2（异构分配）
│   │   ├── (无参考文献)Wang 等 - 2025 - SVD-LLM V2...md
│   │   └── SVDLLMv2.py
│   ├── GFWSVD/                     # 基线：广义 Fisher 加权 SVD
│   ├── DipSVD/                     # 基线：双重重要性保护 SVD
│   ├── ERC-SVD/                    # 基线：误差控制 SVD
│   ├── NSVD/                       # 基线：嵌套激活感知分解
│   ├── FLAT-LLM/                   # 基线：细粒度低秩激活空间变换
│   ├── MGAA/                       # 基线：多粒度自适应分配
│   ├── ProcrustesGPT/              # 基线：正交变换结构化压缩
│   ├── QSVD/                       # 基线：低精度视觉语言模型 SVD
│   ├── ResSVD/                     # 基线：残差补偿 SVD
│   ├── EoRA/                       # 基线：特征空间低秩近似
│   ├── PGSVD/                      # 基线：Pareto 引导低秩压缩
│   └── D-Rank/                     # 基线：层级动态秩压缩
│
├── 论文分析/                        # 对所有相关方法的深度中文分析（44 篇）
│   ├── 三篇基线论文分析.md          # SVD-LLM v1/v2 + GFWSVD 的详细分析
│   ├── 1.论文分析：GFWSVD...md
│   ├── 35.DipSVD...md
│   └── ...（共 44 篇方法分析）
│
├── 推理分析文档/                    # 本文理论推导、算法设计与改进方案
│   ├── 1）公式核对.md               # 所有公式与代码对照核查
│   ├── 2）创新点确认.md             # 本文创新点与基线方法对比
│   ├── 3）理论1证明：Fisher加权截断最优性.md
│   ├── 4）理论2证明：残差补偿的有效性.md
│   ├── 5）理论1完整梳理.md
│   ├── 6）理论2完整梳理.md
│   ├── 7）残差补偿权重旧版证明.md
│   ├── 8）两个理论的完整流程.md
│   ├── 9）完整算法过程-GPT修正版.md
│   ├── SVD-LLM压缩改进方案完整分析文档(GPT 5.2版).md
│   ├── SVD-LLM压缩改进方案完整分析文档(claude Opus 4.5版).md
│   └── 完整算法流程_修正版.md
│
└── README.md                        # 本文件
```

---

## 📄 论文概述

**SAES-SVD** 是一个无需微调的大型语言模型（LLM）低秩压缩框架，核心创新在于：

| 组件 | 全称 | 解决的问题 |
|------|------|-----------|
| **CEALC** | Cumulative Error-Aware Layer Compression | 逐层压缩忽视跨层误差传播，导致全局偏差累积 |
| **ACES** | Adaptive Collaborative Error Suppression | 固定权重系数导致低秩结构次优，能量集中度不足 |

**主要成果**：在 LLaMA-7B、0.2 压缩比条件下，与 Dip-SVD 相比，零样本准确率下降减少 58%，困惑度差距缩小 52%。

---

## 🔍 如何指定文件供 AI 阅读

### 方法一：直接提供文件路径（推荐）

在对话中直接指定路径，例如：

```
请阅读以下文件：
- markdown格式论文/SAES-SVD/Saes-svd_self-adaptive_suppression_of_accumulated_and_local_errors_for_svd-based_llm_compression.md
- 推理分析文档/2）创新点确认.md
- markdown格式论文/SAES-SVD/saes-svd审稿意见.txt
```

### 方法二：按类别提供背景材料

**提供基线论文背景**（让 AI 了解领域知识）：
```
基线方法文件：
1. markdown格式论文/SVD-LLM-v1/(无参考文献)Wang 等 - 2025 - SVD-LLM Truncation-aware singular value d.md
2. markdown格式论文/SVD-LLM-v2/(无参考文献)Wang 等 - 2025 - SVD-LLM V2 Optimizing singular value trun.md
3. 论文分析/三篇基线论文分析.md
```

**修改论文正文**（提供主文件 + 审稿意见）：
```
修改任务：
- 论文正文：markdown格式论文/SAES-SVD/Saes-svd_self-adaptive_suppression_of_accumulated_and_local_errors_for_svd-based_llm_compression.md
- 审稿意见：markdown格式论文/SAES-SVD/saes-svd审稿意见.txt
- 改进方案参考：推理分析文档/SVD-LLM压缩改进方案完整分析文档(GPT 5.2版).md
```

---

## 📋 ICLR 2026 审稿意见摘要

审稿意见完整内容见 [`markdown格式论文/SAES-SVD/saes-svd审稿意见.txt`](markdown格式论文/SAES-SVD/saes-svd审稿意见.txt)

| 审稿人 | 主要意见 | 关键问题 |
|-------|---------|---------|
| **R1** | 缺少计算复杂度分析 | 压缩时间细分；是否与量化结合 |
| **R2** | 仅评估 LLaMA 系列 | Qwen 结果？图 3 数据不一致问题 |
| **R3** | 鲁棒性分析缺失 | 与 AA-SVD 的区别；谱隙分析 |
| **R4** | 基准和模型覆盖有限 | LLaMA 3.1 / Qwen 2.5 结果；代码/数学推理任务 |

---

## 🗂️ 关键文件快速索引

| 需求 | 文件路径 |
|------|---------|
| 论文正文（英文）| `markdown格式论文/SAES-SVD/Saes-svd_self-adaptive_suppression_of_accumulated_and_local_errors_for_svd-based_llm_compression.md` |
| 审稿意见 | `markdown格式论文/SAES-SVD/saes-svd审稿意见.txt` |
| 算法代码 | `markdown格式论文/SAES-SVD/SAES-SVD.py` |
| 理论证明（完整版）| `推理分析文档/8）两个理论的完整流程.md` |
| 改进方案参考 | `推理分析文档/SVD-LLM压缩改进方案完整分析文档(GPT 5.2版).md` |
| 基线分析汇总 | `论文分析/三篇基线论文分析.md` |
| SVD-LLM v1 原文 | `markdown格式论文/SVD-LLM-v1/(无参考文献)Wang 等 - 2025 - SVD-LLM Truncation-aware singular value d.md` |
| SVD-LLM v2 原文 | `markdown格式论文/SVD-LLM-v2/(无参考文献)Wang 等 - 2025 - SVD-LLM V2 Optimizing singular value trun.md` |
| GFWSVD 原文 | `markdown格式论文/GFWSVD/1_Chekalina_等_-_2025_-_Generalized_fisher-weighted_SVD_Scalable_kr.md` |
