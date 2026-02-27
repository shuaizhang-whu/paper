# FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for

# Large Language Model Compression

**Jiayi Tian¹, Ryan Solgi¹, Jinming Lu¹,Yifan Yang¹,Hai Li², Zheng Zhang¹**

1 University of California, Santa Barbara, 2 Intel Corporation

# Abstract

Large Language Models (LLMs) have enabled remarkable progress in natural language pro-cessing, yet their high computational and mem-ory demands pose challenges for deployment in resource-constrained environments. Although recent low-rank decomposition methods offer a promising path for structural compression,they often suffer from accuracy degradation, expen-sive calibration procedures, and result in ineffi-cient modeI architectures that hinder real-world inference speedups. In this paper, we propose FLAT-LLM,a fast and accurate, training-free structural compression method based on fine-grained lowv-rank transformations in the acti-vation space. Specifically, we redluce the hid-den dimension by transforming the weights us-ing truncated eigenvectors computed via head-wise Principal Component Analysis, and em-ploy a greedy budget redistribution strategy to adaptively allocate ranks across decoders. FLAT-LLM achieves efficient and effective weight compression without recovery fine-tuning, which could complete the calibration within a few minutes. Evaluated across 5 mod-els and 11 datasets, FLAT-LLM outperforms structural pruning baselines in generalization and downstream performance, while delivering inference speedups over decomposition-based methods.

# 1 Introduction

Large Language Models (LLMs) have achieved state-of-the-art performance in a wide range of natural language processing and understanding tasks (Bai et al., 2023; Liu et al., 2024; Touvron et al., 2023). However, their substantial parameter counts and computational demands pose signifi-cant challenges for deployment in edge devices and resource-constrained environments. Model compression is a promising direction for reducing

'Code is available in https://github.com/ TTTTTTris/FLAT-LLM.

<!-- 30 LLMPruner 25 SliceGPT 20 SVD-LLM NIXeldled FLAP 15 FLAT-LLM (Ours) 10 5 20 30 40 50 Compression Ratio (%) -->
![](./images/f5cdc9bb2b0da1c0.jpg)

Figure 1: Comparison of WikiText-2 perplexity against various baselines on Llama-2 13B model.

both the model size and computational complex-ity, with popular approaches including quantiza-tion (Huang et al., 2024; Tian et al., 2023),knowl-edge distillation (Sun et al., 2020; Jiao et al.,2020), pruning (Sun et al.,2024;Ma et al.,2023;Yang et al., 2025), and low-rank decomposition (Ashk-boos et al., 2024;Yang et al., 2024a). Among these, low-rank decomposition stands out for its hardware efficiency due to its structural nature.

Singular Value Decomposition (SVD) is a com-mon low-rank decomposition approach for com-pressing the weight matrices of LLMs. How-ever,weights from standard pre-training are of-ten nearly full-rank and difficult to compress (Yu and Wu, 2023). To address this, recent work uses the low-rank nature of activation spaces by pro-jecting weights into these subspaces to improve the compression ratio. For example, ASVD (YYuan et al., 2023) uses activation norms to transform the weights before decomposition, and SVD-LLM (Wang et al., 2024) enhances the accuracy by link-ing truncated singular values to reconstruction loss.

Despite recent improvements, SVD-based meth-ods often incur performance drops and require re-covery fine-tuning, even under low compression ratios. This stems from a key limitation: preserving sufficient information via high-rank SVD increases model parameters, as both left and right singular vectors must be stored. The problem is especially

<!-- S70CInr67[TO.sal CA996s7'SOsZAAIXTe -->

<!-- MHA MLP Wo1 Wo2 WoH W2 multi-head attention SiLU SiLU RoPE RoPE Wq Wk Wv1 Wv2 WvH W1 RMS Norm RMS Norm X Xi Xi+1 -->
![](./images/1dd2f94b980f0ca3.jpg)

<!-- MHA MLP W Wo2 WoH $\tilde {\mathbf {W}}_{2}$ ( multi-head attention SiLU RoPE RoPE Wq Wk Wv1 Wv2 WvH $\tilde {\mathbf {W}}_{1}$ RMS Norm RMS Norm X Xi Xi+1 -->
![](./images/26915791f09557b3.jpg)

Figure 2: Decoder structure before (left) and after (right) weight truncation. Orange blocks indicate truncated weights; hatched areas show removed weights; blue boxes denote non-linear functions.

severe for square matrices (which are common in LLMs like Llama), where reducing parameters re-quires truncating at least 50% of singular values, often leading to significant information loss. More-over, replacing a single large matrix multiplication with two smaller ones in SVD-format linear lay-ers can dlegrade GPU efficiency, limiting practical inference speedups.

To address the inherent limitations of SVD, SliceGPT (Ashkboos et al., 2024) projects post-RMSNorm hidden states into low-rank subspaces using Principal Component Analysis (PCA), en-abling the resulting eigenvector pairs to be fully ab-sorbed into adjacent layers. Furthermore, SliceGPT avoids storing both left and right singular vectors, thereby reducing the information loss and improv-ing the inference efficiency at the same compres-sion ratio. However, to enable this transforma-tion between decoder layers, SliceGPT requires inserting adapter modules along the residual paths, which bring high memory overhead and limit the inference speedup. For example, on Llama-2 7B, SliceGPT incur 10% additional memory overhead at a 20% compression ratio, and yields only 1..1×speedup in inference at a 50% compression ratio.

To further enhance the performance of low-rank compressed models, previous works have proposed rank-adaptive methods that allocate heterogeneous ranks across different model components. For in-stance, adaptive SVD (Gao et al., 2024b) employs a trainable hyper-network for rank selection, achiev-ing better compression than uniform baselines but at the cost of task-specific retraining. However,this method requires days or even weeks to complete for LLLaMA 7B, making it particularly impracti-cal and extremely inefficient for scaling to larger models such as LLaMA 70B.

To overcome the challenges mentioned above, we propose FLAT-LLM, a fast, accurate, and training-free structural compression method for LLMs. Figure 2 ilustrates the decoder architecture before and after applying FLAT-LLM compression. FLAT-LLM projects the post-value hidden states into low-rank subspaces using fine-grained, head-wise PCA.The resulting low-rank eigenvector pairs are then absorbed into the value and output weight matrices to complete the compression. Unlike pre-vious approaches, this joint absorption-based com-pression introduces no additional memory over-head, and the overall compression ratio directly corresponds to the retained rank ratio. To further improve performance, we introduce an importance-preserving rank selection algorithm. This algo-rithm is entirely training-free and completes in just a few seconds, delivering over 100xhigher time efficiency compared to prior rank selection meth-ods such as Adaptive SVD (Gao et al., 2024b). Our main contributions are as follows:

·We propose a training-free,fine-grained compres-sion technique that operates within multi-head attention layers, avoiding the inefficiencies of prior decomposition-based methods.

·We introduce a novel training-free rank selection algorithm that allocates ranks using a greedy re-distribution strategy and can be integrated with existing low-rank LLMI compression pipelines.

·We demonstrate the effectiveness of FLAT-LLM through extensive evaluations on language mod-eling and downstream tasks. As shown in Fig-ure 1, FLAT-LLM significantly improves per-plexity on the WikiText-2 test split across a range of compression ratios, indicating enhanced text generation capabilities under varying levels of compression.

# 2 Related Work

In addition to structural pruning via weight decom-position (as discussed in the introduction), another line of research achieves structural compression by directly removing model components based on im-portance scores. These approaches can be broadly categorized into two types: fine-grained pruning and coarse-grained pruning. The former offers fine-grained control by removing individual rows or columns within weight matrices. For example, LLM-Pruner (Ma et al., 2023) leverages gradient-based saliency scores to prune less important com-ponents, while FLAP (An et al., 2024) adaptively removes unstable neurons and channels based on activation fluctuations.

In contrast, the coarse-grained pruning elimi-nates larger components such as attention heads, layers, or entire decoders. Though more efficient for inference, this often results in greater per-formance degradation. For instance, ShortGPT (Men et al., 2024) prunes decoders based on co-sine similarity-based importance ranking; LaCo (Yang et al.,2024b) compresses models by merg-ing adjacent layers; and BlockPruner (Zhong et al., 2024) removes redundant MHA or MLP blocks through iterative search. In this work, we demon-strate that our method outperforms both fine-and coarse-grained structural pruning baselines in terms of performance under compression.

# 3 Background

**Multi-Head** **Attention.** A typical LLM consists of multiple decoder layers, each comprising two main components: Multi-Head Attention (MHA) and a MMulti-Layer Perceptron (MLP). We define the input hidden statesX∈RN×hi,where N de-note the sequence length, anddhid,dhrepresent the dimension of hidden states and each attention head, respectively.Here, the MHA module computes a weighted aggregation of values using learned pro-jections of queries, keys, and values. For a total of H attention heads, each head h performs:

$$\mathbf {A}^{h}=\frac {\mathbf {X}\mathbf {W}_{q}^{h\top }\left(\mathbf {X}\mathbf {W}_{k}^{h\top }\right)^{\top }}{\sqrt {d_{h}}}\tag{1}$$

$$\mathbf {Y}_{v}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{h\top }$$

whereWqh,WkhWvh∈Rdhxdhidare the projec-tion matrices in query, key and value layer in head h. The attention matrix Ah∈RN×Ncaptures token-wise interactions, and the Softmax function is applied to compute attention scores, which are then used to weight the value representations. The resultYvh∈RNxhrepresents the per-head value output. This is is further transformed by a learned outpout projectionWoh∈Rhixh,yielding the partial attention outputYoh∈RN×dhid.The final output of the multi-head attention layer is thenob-tained by aggregating the partial outputs from all heads:

$$\mathbf {Y}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{h\top }\mathbf {W}_{o}^{h\top }\tag{2}\quad \mathbf {Y}_{o}=\text {sum}\left(\mathbf {Y}_{o}^{1},\cdots ,\mathbf {Y}_{o}^{H}\right)$$

**Principle Component Analysis (PCA).** Princi-pal Component Analysis (PCA) is a classical di-mensioonality reduction technique that identifies the the principal components along which the data ex-hibit the greatest variance. Given a data matrix Z∈RN×d,PCAcomputes the eigen decomposi-tion of the covariance matrix C:

$$\mathbf {C}=\mathbf {Z}^{\top }\mathbf {Z}=\mathbf {Q}\boldsymbol {Λ}\mathbf {Q}^{\top },\tag{3}$$

whereQ∈Rd×ds the orthogonal matrix of eigen-vectors,andΛ∈Rddis the diagonal matrix of corresponding eigenvalues. To capture the most significant variance, we retain the top-r principal components and define $\tilde {\mathbf {Q}}\in \mathbb {R}^{\times }$ as the truncated eigenvector matrix. The corresponding rank-rre-construction of the original data is defined as pro-jecting Z onto therank-r space and then mapping it back to the original space:

$$\tilde {\mathbf {Z}}=\mathbf {Z}\tilde {\mathbf {Q}}\tilde {\mathbf {Q}}^{T}.\tag{4}$$

PCA thus reveals low-rank structure in weight ma-trices or hidden representations, enabling efficient approximations with minimal loss of information.

# 4 FLAT-LLM

In this section, we present FLAT-LLM in detail, be-ginning with a head-wise PCA-based weight trun-cation method that compresses weight matrices by truncating and absorbing the eigenvectors. We then introduce an importance-preserving rank selection strategy to allocate adaptive ranks across decoder layers. Finally, we conduct a theoretical analysis of the truncation loss in our head-wise PCA method.

## 4.1 Head-wise PCA-based Weight Truncation

Inspired by Equation (2), we observe that the value and output projections within each attention head

<!-- Qv1 Qv2 --- QvH ② Truncation Q $\tilde {\mathbf {Q}}_{v}^{1}$ Q2 $\tilde {\mathbf {Q}}_{v}^{H}$ Qv1\top Qv2\top -.. QvH\top $\tilde {\mathbf {Q}}_{1}^{1}$ $\tilde {\mathbf {Q}}_{v}^{2}$ -.. QvH ① PCA ③ Linear Mapping Cv1 Cv2 CvH $\tilde {\mathbf {W}}_{v}^{h}=\tilde {\mathbf {Q}}_{v}^{h}{}^{\top }\mathbf {W}_{v}^{h}$ Wv1 Wv2 -.. WvH $\tilde {\mathbf {W}}_{v}^{1}$ Wv2 ... $\tilde {\mathbf {W}}$ w! -->
![](./images/fc69cce722724cd1.jpg)

Figure 3: Fine-grained head-wise PCA in value layer.

are computed consecutively. Leveraging this obser-vation, we propose to exploit the low-rank activa-tion space of the value output to jointly compress the value and output weight matrices.

As shown in Figure 3, the detail compression process with M calibration samples are given in three steps: ① Compute the covariance matrix Cvh=∑m=1MYv,mh\topYv,mhand perform PCA to obtain the orthogonal eigenvectorQvh∈Rh×h utilizing Equation (3). ② Truncate the eigenvec-tors to rank r, yielding reduced basis $\tilde {\mathbf {Q}}_{v}^{h}\in \mathbb {R}^{d_{h}\times r},$ and the reconstructed per-head value output be-comes $\tilde {\mathbf {Y}}_{v}^{h}=\mathbf {Y}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }$ .③ To compress the weights by absorbing the truncated eigenvectors, we reformulate the MHA computation in Equation (2) as the following:

$$\mathbf {Y}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{h\top }\mathbf {Q}_{v}^{h}\mathbf {Q}_{v}^{h\top }\mathbf {W}_{o}^{h\top }\tag{5}$$

This enables the jointly compression on the value and output weights using the PCA basis derived from the value layer:

$$\tilde {\mathbf {Y}}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{h\top }\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }\mathbf {W}_{o}^{h\top }\quad \tilde {\mathbf {Y}}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\tilde {\mathbf {W}}_{v}^{h\top }\tilde {\mathbf {W}}_{o}^{h\top }\tag{6}$$

where the first and second equations represent the truncation and absorption. Here, we aim for $\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }\approx \mathbf {I}_{v}^{h}$ to retain most of the representa-tional power of the original Yvh. After absorb-ing the truncated basis into the weights, the value and output projections of each head are reduced to $\tilde {\mathbf {W}}_{v}^{}\in \mathbb {R}^{r\times _{i}},\tilde {\mathbf {W}}_{o}^{}\in \mathbb {R}^{_{i}\times r}$ ,respectively.

In this way, we can jointly compress both the value and output weights to $\frac {r}{d_{h}}$ of their original size leveraging the low-rank structure of the output hidden states from the value layer. Notably,this joint compression technique remains compatibole

### Algorithm 1 Importance-Preserving Rank Selec-tion

**Require**: Sparsity S, number of decoders L, im-portance scores t

1: Initialize total budgetB←L(1-s),active setA←{1,⋯,L}

2**:while**A≠\emptysetdo

3:

$\tilde {w}_{l}=\frac {t_{l}}{\sum _{j\in \mathcal {A}}t_{j}}$ ·B foralll∈A

4:

$$\text {Let}\mathcal {S}\leftarrow \left\{l\in \mathcal {A}|\tilde {w}_{l}>1\right\}$$

5:

**if**S=\emptyset**then** Δ Assign all remain entries

6:

$w_{l}\leftarrow \tilde {w}_{l}$ for alll∈A

7:

**break**

8:

**end if**

9:

**for all**l∈Sdo

Δ Assign fixed entries

10:

$$w_{l}\leftarrow 1$$

11:

$$B\leftarrow B-w_{l}$$

12:

**end for**

13:

$$\mathcal {A}\leftarrow \mathcal {A}\backslash \mathcal {S}$$

Δ Remove fixed entries

14**: end while**

15**: return w**

Final allocation

with modern architectures using Grouped-Query Attention (GQA) (Ainslie et al., 2023), resulting in different numbers of value and output heads.The detailed formulation for the GQA case is provided in Appendix A.

## 4.2 Importance-Preserving Rank Selection

In our experiments, we observed that using a uni-form rank across all layers degrades performance, particularly under high compression ratios. We first analyze the cosine similarity between each decoder layer's input and output hidden states, and reveal that the intrinsic dimensionality varies across lay-ers. Motivated by this observation, we propose a decoder-wise rank selection algorithm that em-ploys a greedy redistribution strategy to adaptively allocate ranks based on their relative importance.

To analyze variations in intrinsic dimension across decoder layers, we compute the cosine sim-ilarity between the input and output hidden states of each decoder. Given a model with L decoder layers,letXlandXl+1denote the input and out-put hidden state matrices of the l-th decoder layer, respectively.For each samplep∈{1,⋯,N},the cosine similaritycl,pbetween the corresponding rows ofXlandXl+1is defined as:

$$c_{l,p}=\frac {\mathbf {X}_{l,p}{}^{\top }\mathbf {X}_{l+1,p}}{\left\|\mathbf {X}_{l,p}\right\|_{2}\left\|\mathbf {X}_{l+1,p}\right\|_{2}},\tag{7}$$

whereXl,pandXl+1,p represent the p-th row of

XlandXl+1,respectively. The average cosine similarity for the l-th decoder layer is then given bycl=EX,p[cl,p],reflecting the overall alignment between input and output hidden states.

To quantify the degree of dissimilarity and thereby infer the intrinsic dimension, we compute $t_{l}=\frac {\arccos c_{l}}{\pi }$ , which captures the normalized an-gular deviation between the representations. We interprettl as an indicator of the relative impor-tance and compressibility of each decoder layer. From empirical evaluations on several LLMs, we observe thattl typically ranges from 0.06 to 0.30. This leads to two key insights: (1) intrinsic dimen-sionality varies across layers, motivating the use of heterogeneous rank assignments; and (2) the small angular deviations suggest that decoder layers of-ten reside in low-dimensional subspaces, indicating that LLMs are amenable to compression.

Given the model sparsity as S, to determine the remaining rank ratios wl for l-th decoder regard-ing the importance scoretl,, a naive way is to pro-portional scale the total budget of remaining rank ratiosB=L(1-s),which gives\hat{w}_{l}=\frac{t_{l}}{\sum_{l=1}^{L}t_{l}}B. However, this naive proportional allocation may violate thewl∈[0,1]constraint when some com-ponents receive disproportionately high scores. In order to design wl regarding the importance score tl, while fit each wli within the constraint and fix the total budget for remaining size as∑w=B, the objective of our rank selection algorithm can be defined as:

$$\min _{\mathbf {w}\in [0,1]^{L}}\|\mathbf {w}-\hat {\mathbf {w}}\|,\text {s.t.}\sum _{l}w_{l}=B,\tag{8}$$

wherew=[w1,⋯,wL]and $\hat {\mathbf {w}}=\left[\hat {w}_{1},\cdots ,\hat {w}_{L}\right]$ are vectors constructed by the ratios wl and $\hat {w}_{l}$  in each decoders.

To address this, we immplement a greedy redistri-bution strategy, as shown in Algorithm 1. First, we define a variable $\tilde {w}_{l}$  to represent the proportional scaled ratios $\hat {w}_{l}$  when the budgetB is changing dur-ing the rank selection process. Given an active set A that contains the indices of unassigned remain-ing rank ratios, we iteratively compute $\tilde {w}_{l}$  in the active set with latest budget B, clip $\tilde {w}_{l}$ thatexceed the upper bound as 1, update the total budget B,and remove the clipped entries from the active set. This process continues until all elements has been assigned a remaining rank ratios wl. In this way, the resulting solution wl remains proportional to tl where possible, while ensuring the boundedness and budget feasibility.

<!-- 100 100 80 80 (3)oHeN YUe 6uIUIeue 60 60 40 40 20 20 (8)oeN 6uIUIeueN 6Ny 0 0 5 10 15 20 25 30 Layer ID -->
![](./images/195c521cf2034e42.jpg)

Figure 4: Remaining rank ratio versus layer id computed with Algorithm 1. The average remaining ratio is set between 30% (lowest solid) to 90% (highest solid).

Figure 4 illustrates the remaining rank ratios across the decoder blocks of Llama-2 7B under different average remaining ratios (from 40% to 90%) determined by the IPRS algorithm. The gray dashed curve represents the normalized importance scores t that guide the optimization process,while the colorful solid lines show the remaining rank ratio w under multiple sparsity ratios∈[0.1,0.6]. As shown, the resulting rank allocations are non-uniform and adaptively preserve more capacity in layers with higher importance. Additional visual-izations of the rank distributions produced by IPRS on other models are provided in Appendix D.3.

## 4.3 Truncation Error Analysis

In the following, we provide the theoretical proof on the direct mapping between eigenvalues and truncation loss in single-head and multi-head cases, which guarantee the effectiveness of our method. The detailed pproofs are shown in Appendix B.

As described in Section 4.1, to compress the value layer, we perform PCA on its feature covari-ance and project onto a low-rank subspace:

$$\tilde {\mathbf {Y}}_{v}^{h}=\mathbf {Y}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }\tag{9}$$

where $\mathbf {Y}_{v}^{h\top }\mathbf {Y}_{v}^{h}=\mathbf {Q}_{v}^{h}\boldsymbol {Λ}_{v}^{h}\mathbf {Q}_{v}^{h\top },\tilde {\mathbf {Q}}_{v}^{h}=\mathbf {Q}_{v}^{h}[:,:r]$ 

Here,Qh∈Rdh×dhcontains orthonormal eigen-vectors andΛhis diagonal with eigenvaluesλ1h≥∴≥λd≥0, and $\tilde {\mathbf {Q}}^{}\in \mathbb {R}^{d_{}\times r}$ contains the top-r principal components.

**Theorem** **4.1** (Reconstruction Error of Single-head Output PCA Projection).LetYvh=XWvh\top,and $\tilde {\mathbf {Y}}_{v}^{h}=\mathbf {Y}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }$  be the rank-r approximation obtained by projectingYvhonto its top-r principal components. Then the squared Frobenius norm of the reconstruction error satisfies:

$$\left\|\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right\|_{F}^{2}=\sum _{i=r+1}^{d_{h}}λ_{i}^{h}$$


![](./images/00a1df8475171433.jpg)

where{λih}are the eigenvaluesofYvh\topYvh. **Corollary** **4.2** (Reconstruction Error of Mul-ti-head Output PCA Projection).LeYv= concat((Yv1,⋯YvH)be the concatenated output hid-den states. The squared Frobenius norm of the reconstruction error satisfies:

$$\|\mathbf {Y}_{v}-\tilde {\mathbf {Y}}_{v}\|_{F}^{2}=\sum _{h=1}^{H}\sum _{i=r+1}^{d_{h}}\lambda _{i}^{h},$$

Therefore, the reconstruction error of the multi-head value output can be expressed as the sum of the truncated eigenvaluesλihfrom the output of each head value projection Yvh.When the preserved dimension is r, truncating the smallest eigenvaluesλr+1h..λhin the PCA decomposition of each value head yields the minimal reconstruc-tion error for the multi-head value output.

# 5 Experiments

In this section, we first compare the performance of FLAT-LLM on language modeling and dowvn-stream tasks with recent fine-grained structural pruning methods across multiple LLM architec-tures and compression ratios. We then evaluate inference speedup and memory saving, and con-duct ablation studies on our IPRS algorithm and the impact of calibration datasets. Additional ex-perimental results, including calibration efficiency, performance under varying compression ratios on downstream tasks, and comparisons with coarse-grained importance-based pruning baselines, are provided in Appendix D.

## 5.1 Setups

**Models** **and** **Datasets.** We evaluate our method on multiple decoder-based generative models, in-cluding Llama-2 7B, 13B,70B,Llama-3 8B (Tou-vron et al., 2023), and Mistral 7B-v0.1 (Jiang et al., 2023). Following the settings in previ-ous works (Ashkboos et al., 2024;Wang et al., 2024), we use 256 samples with 4096 tokens from WikiText-2 or Alpaca datasets for calibration. For the downstream evaluation, we use the LM Eval-uation Harness (Gao et al., 2024a) and test on ten tasks: ARC-e, ARC-c, PIQA,WinoGrande, HellaSwag, BoolQ, OBQA, MathQA, Common-senseQA,MMLU. Here, MMLU uses 5-shot eval-uation, and all others use zero-shot.

**Baselines.** We compare our method with recent decomposition-based pruning approaches, includ-ing SVD-LLM (Wang et al., 2024) and SliceGPT (Ashkboos et al., 2024), as well as fine-grained importance-based methods such as FLAP (An et al., 2024) and LLM-Pruner (Ma et al., 2023). Addi-tionally, we evaluate our approach against coarse-grained structural pruning techniques, including ShortGPT (Men et al., 2024),LaCo (Yang et al., 2024b), and BlockPruner (Zhong et al., 2024),as reported in Appendix D.2.

**Implementation Details.** Our implementation is built on the Huggingface Transformers library (Wolf et al., 2019). The MHA blocks are com-pressed using our proposed head-wise PCA, while the MLP modules are compressed using selection matrices derived from the Nyström approximation; implementation details are provided in Appendix C. All decomposition computations are performed in double precision to ensure numerical1 stability.Ex-periments are conducted on a single A100 40GB GPU,except for LLaMA-2 70B,which is evaluated using 4 A100 GPUs. All methods are evaluated without any recovery fine-tuning.

## 5.2 Performance Comparison

**Performance on Different LLMs.** Table 1 com-pares the language modeling perplexity and down-stream task accuracy of FLAT-LLM against prior structural compression methods across five LLMs under a 20% compression ratio. As noted in prior work (Huang et al., 2025), LLM-Pruner is incom-patible wvith grouped-query attention architectures, and therefore results are omitted for LLaMA-2-70B,LLaMA-3-8B, and Mistral-7B. Across all evaluated models, FLAT-LLM achieves the best overall performance in both average accuracy and perplexity,indicating strong generalization and text generation quality under compression. For larger models, like LLaMA-2 13B and 70B, FLAT-LLM incurs only a modest average accuracy drop of 2.7% and 3.4%. In contrast to prior methods, which of-ten struggle on newer architectures like Mistral-7B and LLaMA-3 8B, FLAT-LLM consistently main-tains high performance, yielding up to 15.4% and 20.2% accuracy improvements, respectively. NNo-tably,FLAT-LLM outperforms all baselines by as much as 40% on the MMLU 5-shot benchmark, demonstrating its effectiveness at preserving fac-tual reasoning and broad-domain knowledge.

We further evaluate zero-shot downstream per-

Table 1: Comparison of downstream performance against prior structural compression methods on LLaMA-2 (7B, 13B,70B),LLaMA-3 8B, and Mistral-7B models models at 20% compression ratio.

<table border="1" ><tr>
<td>Model</td>
<td>Method</td>
<td>PPL↓</td>
<td>Avg.↑</td>
<td>MMLU 5-shot</td>
<td>PIQA</td>
<td>WinoG.</td>
<td> HellaS.</td>
<td>ARC-e</td>
<td>ARC-c</td>
<td>OBQA</td>
</tr><tr>
<td rowspan="6">LLaMA-2 7B</td>
<td>Original</td>
<td>5.11</td>
<td>62.14</td>
<td>45.70</td>
<td>79.05</td>
<td>69.38</td>
<td>75.92</td>
<td>74.49</td>
<td>46.25</td>
<td>44.20</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>10.55</td>
<td>53.89</td>
<td>26.20</td>
<td>75.95</td>
<td>63.38</td>
<td>67.83</td>
<td>64.31</td>
<td>39.93</td>
<td>39.60</td>
</tr><tr>
<td>FLAP</td>
<td>6.76</td>
<td>53.07</td>
<td>31.90</td>
<td>74.54</td>
<td>62.98</td>
<td>64.74</td>
<td>61.28</td>
<td>36.43</td>
<td>39.60</td>
</tr><tr>
<td>SliceGPT</td>
<td>8.24</td>
<td>46.26</td>
<td>26.75</td>
<td>64.80</td>
<td>62.98</td>
<td>49.18</td>
<td>55.68</td>
<td>31.40</td>
<td>33.00</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.84</td>
<td>52.32</td>
<td>29.34</td>
<td>71.49</td>
<td>65.27</td>
<td>58.57</td>
<td>68.31</td>
<td>35.84</td>
<td>37.40</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>6.70</td>
<td>55.16</td>
<td>39.67</td>
<td>72.20</td>
<td>65.82</td>
<td>64.72</td>
<td>64.44</td>
<td>38.65</td>
<td>40.60</td>
</tr><tr>
<td rowspan="6">LLaMA-2 13B</td>
<td>Original</td>
<td>4.57</td>
<td>65.70</td>
<td>55.40</td>
<td>80.41</td>
<td>72.53</td>
<td>79.41</td>
<td>77.39</td>
<td>49.15</td>
<td>45.60</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>9.67</td>
<td>55.45</td>
<td>22.80</td>
<td>77.97</td>
<td>60.77</td>
<td>71.26</td>
<td>67.09</td>
<td>44.28</td>
<td>44.00</td>
</tr><tr>
<td>FLAP</td>
<td>5.90</td>
<td>57.00</td>
<td>41.20</td>
<td>75.57</td>
<td>67.25</td>
<td>69.19</td>
<td>65.91</td>
<td>39.08</td>
<td>40.80</td>
</tr><tr>
<td>SliceGPT</td>
<td>7.10</td>
<td>50.58</td>
<td>35.49</td>
<td>65.18</td>
<td>65.67</td>
<td>52.30</td>
<td>59.26</td>
<td>36.77</td>
<td>39.40</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.37</td>
<td>55.86</td>
<td>35.54</td>
<td>72.91</td>
<td>67.17</td>
<td>63.47</td>
<td>71.00</td>
<td>39.93</td>
<td>41.00</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>5.55</td>
<td>63.00</td>
<td>54.72</td>
<td>75.84</td>
<td>72.06</td>
<td>73.36</td>
<td>75.59</td>
<td>46.25</td>
<td>43.20</td>
</tr><tr>
<td rowspan="5">LLaMA-2 70B</td>
<td>Original</td>
<td>3.12</td>
<td>71.38</td>
<td>68.80</td>
<td>82.75</td>
<td>77.82</td>
<td>83.80</td>
<td>80.72</td>
<td>57.17</td>
<td>48.60</td>
</tr><tr>
<td>FLAP</td>
<td>8.76</td>
<td>48.29</td>
<td>25.90</td>
<td>72.31</td>
<td>64.09</td>
<td>55.94</td>
<td>51.05</td>
<td>31.91</td>
<td>36.80</td>
</tr><tr>
<td>SliceGPT</td>
<td>5.76</td>
<td>57.40</td>
<td>48.30</td>
<td>68.01</td>
<td>72.14</td>
<td>57.16</td>
<td>68.64</td>
<td>43.94</td>
<td>43.60</td>
</tr><tr>
<td>SVD-LLM</td>
<td>5.96</td>
<td>61.07</td>
<td>52.10</td>
<td>74.48</td>
<td>72.61</td>
<td>68.41</td>
<td>71.93</td>
<td>46.93</td>
<td>41.00</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>4.33</td>
<td>67.98</td>
<td>67.35</td>
<td>77.20</td>
<td>77.03</td>
<td>78.44</td>
<td>78.87</td>
<td>51.19</td>
<td>45.80</td>
</tr><tr>
<td rowspan="5">Mistral-7B</td>
<td>Original</td>
<td>4.92</td>
<td>68.14</td>
<td>62.50</td>
<td>82.05</td>
<td>73.95</td>
<td>81.02</td>
<td>79.55</td>
<td>53.92</td>
<td>44.00</td>
</tr><tr>
<td>FLAP</td>
<td>7.11</td>
<td>48.29</td>
<td>25.90</td>
<td>72.31</td>
<td>64.09</td>
<td>55.94</td>
<td>51.05</td>
<td>31.91</td>
<td>36.80</td>
</tr><tr>
<td>SliceGPT</td>
<td>9.06</td>
<td>43.18</td>
<td>25.52</td>
<td>59.35</td>
<td>61.21</td>
<td>45.11</td>
<td>51.60</td>
<td>30.29</td>
<td>29.20</td>
</tr><tr>
<td>SVD-LLM</td>
<td>9.29</td>
<td>50.23</td>
<td>25.02</td>
<td>70.46</td>
<td>64.56</td>
<td>58.09</td>
<td>69.28</td>
<td>37.63</td>
<td>26.60</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>6.11</td>
<td>58.54</td>
<td>57.21</td>
<td>70.84</td>
<td>69.30</td>
<td>63.76</td>
<td>68.73</td>
<td>41.72</td>
<td>38.20</td>
</tr><tr>
<td rowspan="5">LLaMA-3 8B</td>
<td>Original</td>
<td>5.75</td>
<td>68.14</td>
<td>65.43</td>
<td>80.85</td>
<td>73.40</td>
<td>79.17</td>
<td>80.09</td>
<td>53.24</td>
<td>44.80</td>
</tr><tr>
<td>FLAP</td>
<td>8.42</td>
<td>54.42</td>
<td>42.24</td>
<td>73.50</td>
<td>65.90</td>
<td>59.87</td>
<td>64.22</td>
<td>36.26</td>
<td>39.00</td>
</tr><tr>
<td>SliceGPT</td>
<td>16.62</td>
<td>41.28</td>
<td>25.07</td>
<td>60.23</td>
<td>57.22</td>
<td>40.46</td>
<td>47.26</td>
<td>27.73</td>
<td>31.00</td>
</tr><tr>
<td>SVD-LLM</td>
<td>17.17</td>
<td>47.06</td>
<td>28.64</td>
<td>66.27</td>
<td>61.01</td>
<td>52.65</td>
<td>55.43</td>
<td>31.66</td>
<td>33.80</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>8.15</td>
<td>61.47</td>
<td>62.56</td>
<td>72.80</td>
<td>72.53</td>
<td>69.13</td>
<td>68.90</td>
<td>42.58</td>
<td>41.80</td>
</tr></table>

formance under varying compression ratios (see Figure 8 in Appendix D.2). FLAT-LLM consis-tently outperforms prior low-rank decomposition baselines across all models and compression levels, demonstrating strong robustness and effectiveness.

## 5.3 Comparison with Quantization

We further integrate FLAT-LLM with post-training quantization methods to assess their combined ef-fectiveness. As shown in Table 2, applying FLAT-LLM (with a 30% compression ratio) followed by GPTQ-3/4-bit quantization achieves loower perplex-ity than GPTQ-2/3-bit alone, while maintaining the same or even lower memory footprint. This integration makes it possible to compress large lan-guage models by5-8×Wvith negligible accuracy degradation, enabling highly efficient and effective model deployment.

## 5.4 Inference Efficiency

Figure 5 compares the inference throughput and memory usage of our method with prior low-rank approaches, including SliceGPT and SVD-LLM, across compression ratios ranging from 10% to 50% on LLaMA-2 7B. Following previous setups

Table 2: Perplexity comparison with GPTQ for LLaMA-2 7B and Llama-3 8B on WikiText-2.

<table border="1" ><tr>
<td>Model</td>
<td>Method</td>
<td>Size (GB)</td>
<td>PPL↓</td>
</tr><tr>
<td rowspan="3">LLaMA-2 7B</td>
<td>Original</td>
<td>14</td>
<td>5.12</td>
</tr><tr>
<td>GPTQ 2-bit</td>
<td>1.8(7.8×)</td>
<td>NaN</td>
</tr><tr>
<td>FLAT-LLM<br>+GPTQ 3-bit</td>
<td>1.8(7.8×)</td>
<td>13.43</td>
</tr><tr>
<td rowspan="3">LLaMA-3 8B</td>
<td>Original</td>
<td>16</td>
<td>5.75</td>
</tr><tr>
<td>GPTQ 3-bit</td>
<td>3.0(5.3×)</td>
<td>39.85</td>
</tr><tr>
<td>FLAT-LLM<br>+GPTQ 4-bit</td>
<td>2.8(5.7×)</td>
<td>11.91</td>
</tr></table>

(Ashkboos et al., 2024), we evaluate with a batch size of 128 and a sequence length of 256. Without any CUDA-level optimization, our method con-sistently outperforms the original model, achiev-ing speedups ranging from 1.04×to1.23×athe compression ratio increases. Additionally, due to the reduced dimensionality of the value outputs, FLAT-LLM further reduces activation memory in the value cache. As a result, even under the same compression ratio, it achieves the lowest mem-ory usage among all low-rank model compression methods. Compared to SliceGPT and SVD-LLM, it achieves up to 20% and 44% higher throughput, and reduces memory usage by 14% and 16%,re-

<!-- -- Original (Speedup) Original (Memory) FLAT-LLM (Speedup) FLAT-LLM (Memory) Slicegpt (Speedup) Slicegpt (Memory) SVD-LLM (Speedup) SVD-LLM (Memory) 3000 ·····.. 30 (S/suexOL) -2 9.7% 25 2500 -17.6% -24.8% 20 -32.8% 1.23x 2000 1.17x 15 IndubnoAL e0uae드I 1.10x 1.11x 。 x 1.04× 。。 10 1500 。 。 。 。 。 00 0 。0 00 (89)a6esn AoueW 。 。 。 。 。 5 。 。0 。0 。。 0 0 1000 30% 50% 0 10% 20% 40% Compression Ratio -->
![](./images/599ad3d67e444698.jpg)

Figure 5: Comparison of inference throughput and mem-ory usage with prior low-rank-based methods.

<!-- 60 (%) 02h -62y 50 Original IPRS (Alpaca) Uniform (Alpaca) IPRS (WikiText-2) 40 Uniform(WikiText-2) 10% 20% 30% 40% Compression Ratio -->
![](./images/7c89d2fb2c1cf1db.jpg)

Figure 6: Comparison of zero-shot average accuracy on downstream datasets versus compression ratio using uniform rank and our IPRS algorithm on Llama-2 13B.

spectively. These results highlight the efficiency of FLAT-LLM, making it a strong candidate for practical,real-world deployment.

## 5.5 Ablation Study

In this section, we present an ablation study of our rank selection method and evaluate its perfor-mance using different calibration datasets. Specifi-cally,we report the average accuracy across eight downstream tasks-ARC-e,ARC-c,PIQA,Wino-Grande, HellaSwag, BoolQ, MathQA, and CCom-monsenseQA-using calibration performed on1ei-ther the WikiText-2 or Alpaca dataset.

**Importance-Preserving Rank Selection.** To evaluate the effectiveness of our Importance-Preserving Rank Selection (IPRS) method, Fig-ure 6 compares the average zero-shot precision of different rank allocation strategies - uniform and IPRS - between compression ratios ranging from 10% to 40% on LLaMA-2 13B. As shown in Fig-ure 6, IPRS consistently outperforms the uniform baseline across all compression settings. The per-formance gap becomes more pronounced at higher compression ratios, achieving gains of 5.4% with

<!-- FLAT-LLM (Ours) Slicegpt SVD-LLM Original WikiText-2 Alpaca 65 65 60 60 55 9 55 (3)aoy -BNy 50 50 45 6Ng 45 40 40 35 10% 20% 30% 40% 35 10% 20% 30% 40% Compression Ratio Compression Ratio -->
![](./images/1f7f3b34e2cebc62.jpg)

Figure 7: Comparison of zero-shot average accuracy on downstream datasets versus compression ratio cali-brated on WikiText-2 or Alpaca on Llama-2 13B.

Alpaca calibration and 8.0% with WikiText-2 at a 40% compression ratio. This is because sensi-tive layers suffer disproportionately high truncation loss under aggressive compression, making adap-tive rank selection increasingly important. These results underscore the complementary strengths of fine-grained PCA and importance-aware rank allo-cation. In Appendix D.4, we further apply IPRS to the SVD-LLM method, which also shows im-proved performance over uniform rank.

**Calibration** **Dataset.** We evaluate the average zero-shot accuracy of FLAT-LLM on LLaMA-213B using two calibration datasets-WikiText-2and Alpaca-to assess its generality. As shown in Figure 7, FLAT-LLM consistently outperforms SliceGPT and SVD-LLM across all settings and compression levels, demonstrating strong gener-alization with WikiText-2 calibration and even greater gains with Alpaca. Notably, under Alpaca calibration, FLAT-LLM maintains high accuracy, with less than a 5% drop observed at up to 30% compression. These results highlight the robust-ness of FLAT-LLM across diverse tasks, calibration datasets, and compression regimes.

# 6 Conclusion

We propose a fast and accurate training-free struc-tural compression method for LLMs, leveraging low-rank transformations in the activation space. By combining an importance-preserving global rank allocation strategy with efficient head-wise PCA-based approximations, our approach delivers model with strong generation performance using minimal calibration time. Experiments on LLaMA and Mistral show superior generalization and sub-stantial inference speedups compared to prior struc-tural pruning and decomposition-based methods.

## Limitations

While FLAT-LLM demonstrates strong empirical performance in compressing large language mod-els without fine-tuning, several limitations remain. First, the current approach targets the compres-sion of value and output projections in attention layers via head-wise PCA and joint compression, but this strategy does not readily extend to the query and key projections. Second, although FLAT-LLM achieves superior performance and reduced memory consumption when combined with post-training quantization methods such as GPTQ, fur-ther CUDA kernel optimization is necessary to re-alize actual speedup in practice. We leave both of these challenges for future work.

# References

Joshua Ainslie, James Lee-Thorp,Michiel de Jong,Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query trans-former models from multi-head checkpoints. In Pro-ceedings of the 2023 Conference on Empirical Meth-ods in Natural Language Processing, pages 4895-4901.

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. 2024. Fluctuation-based adaptive structured pruning for large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 10865-10873.

Saleh Ashkboos, Maximilian L Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. In The Twelfth Interna-tional Conference on Learning Representations.

Jinze Bai,Shuai Bai, Yunfei Chu,Zeyu Cui,Kai Dang, Xiaodong Deng,Yang Fan, Wenbin Ge,Yu Han,Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Bider-man,Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds,Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024a. A framework for few-shot language model evaluation.

Shangqian Gao, Ting Hua, Yen-Chang Hsu,Yilin Shen, and Hongxia Jin. 2024b. Adaptive rank selections for low-rank approximation of language models. In Proceedingsofthe2024Conferenceof the North American Chapter of the Association for Computa-tional Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 227-241.

Wei Huang,Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, and Xiaojuan Qi. 2024. Billm: pushing the limit of post-training quantization for llms. In Proceedings of the 41st International Conference on Machine Learning, pages 20023-20042.

Xinhao Huang, You-Liang Huang, and Zeyi Wen. 2025. Sola: Leveraging soft activation sparsity and low-rank decomposition for large language model com-pression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 17494-17502.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-sch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand,Gianna Lengyel, Guil-laume Lample,Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen,Linlin Li, Fang Wang, and Qun Liu. 2020. Tinybert: Distilling bert for natural language under-standing. In Findings of the Association for Computa-tional Linguistics: EMNLP 2020,pages 4163-4174.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 20)23. Llm-pruner: On the structural pruning of large lan-guage models. Advances in neural information pro-cessing systems, 36:21702-21720.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han,and Weipeng Chen. 2024. Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. 2024. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang,and Denny Zhou. 2020. Mobilebert:a compact task-agnostic bert for resource-limited de-vices. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics,pages 2158-2170.

Jiayi Tian, Chao Fang, Haonan Wang, and Zhongfeng Wang.2023. Bebert: Efficient and robust binary ensemble bert. In ICASSP 2023-2023 IEEE Interna-tional Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1-5. IEEE.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and effi-cient foundation language models. arXiv preprint arXiv:2302.13971.

Xin Wang,Yu Zheng,Zhongwei Wan, and Mi Zhang. 2024. Svd-llm: Truncation-aware singular value de-composition for large language model compression. arXiv preprint arXiv:2403.07378.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,Clement Delangue, Anthony Moi, Pier-ric Cistac, Tim Rault, Rémi Louf, Morgan Funtow-icz, and 1 others. 2019. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771.

Yifan Yang, Kai Zhen, Bhavana Ganesh, Aram Gal-styan, Goeric Huybrechts, Markus Müller, Jonas M Kübler, Rupak Vignesh Swaminathan, Athanasios Mouchtaris, Sravan Babu Bodapati, and 1 others. 2025. Wanda++: Pruning large language models via regional gradients. arXiv preprint arXiv:2503.04992.

Yifan Yang, Jiajun Zhou, Ngai Wong, and Zheng Zhang. 2024a. Loretta: Low-rank economic tensor-train adaptation for ultra-low-parameter fine-tuning of large language models. In Proceedings of the 2024Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3161-3176.

Yifei Yang,Zouying Cao, and Hai Zhao. 2024b. Laco: Large language model pruning via layer collapse. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 6401-6417.

Hao Yu and Jianxin Wu. 2023. Compressing transform-ers: features are low-rank, but weights are not! In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11007-11015.

Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu,Yan Yan,and Guangyu Sun. 2023. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821.

Longguang Zhong, Fanqi Wan, Ruijun Chen, Xiaojun Quan, and Liangzhi Li. 2024. Blockpruner: Fine-grained pruning for large language models. arXiv preprint arXiv:2406.10594.

# A Head-wise PCA forGroupedQuery Attention

**Grouped Query Attention.** Modern architec-tures such as Llama-3 and Mistral employ Grouped-Query Attention (GQA), in which multiple query heads share a smaller set of key-value heads. This design aims to improve the inference efficiency by reducing both the memory footprint and computa-tion cost associated with the KV cache.

Let H and G denote the number of query and key-value heads,where typicallyH&gt;G.For each query headh∈{1,⋯,H},let its associated key-value head be denoted byg(h)∈{1,⋯,G}.In most cases,whereH=nG for someinteger n,the mapping is defined as $(h)=\left\lfloor \frac {h}{n}\right\rfloor$ . Under this setting,we reformulate Equation (1) as follows:

$$\mathbf {A}^{h}=\frac {\mathbf {X}\mathbf {W}_{q}^{h\top }\left(\mathbf {X}\mathbf {W}_{k}^{g(h)\top }\right)^{\top }}{\sqrt {d_{h}}}\tag{10}$$

$$\mathbf {Y}_{v}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{g(h)\top }$$

where the key difference lies inthe use of shared key and value projectionWkg(h),Wvg(h)across mul-tiple query heads. Similarly, Equation (2) can be rewritten as:

$$\mathbf {Y}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{g(h)\top }\mathbf {W}_{o}^{h\top }.\tag{11}$$

**Head-wise** **PCCA** **for** **GQA** Although GQA intro-duces a mismatch between the number of heads in the value and output layers, the joint compres-sion technique remains applicable to GQA-based architectures.

LetYvg(h)represent the output of the value layer for key-value headg(h).By applying PCA,we can transform the value output as

$$\mathbf {Y}_{v}^{g(h)}=\mathbf {Y}_{v}^{g(h)}\mathbf {Q}_{v}^{g(h)}\mathbf {Q}_{v}^{g(h)^{\top }}$$

and reformulate Equation (11)as

$$\mathbf {Y}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{g(h)\top }\mathbf {Q}_{v}^{g(h)}\mathbf {Q}_{v}^{g(h)\top }\mathbf {W}_{o}^{h\top }$$

Even though the output layer contains H heads and the value layer contains only G, each output projection head h uses the PCA basis Qvg()de-rived from its corresponding value head(h).This enables joint compression of both the value and output projection layers under GQA. The process can be expressed as:

$\tilde {\mathbf {Y}}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\mathbf {W}_{v}^{g(h)\top }\tilde {\mathbf {Q}}_{v}^{g(h)}\tilde {\mathbf {Q}}_{v}^{g(h)\top }\mathbf {W}_{o}^{h\top }$ $\tilde {\mathbf {Y}}_{o}^{h}=\text {Softmax}\left(\mathbf {A}^{h}\right)\mathbf {X}\tilde {\mathbf {W}}_{v}^{g(h)\top }\tilde {\mathbf {W}}_{o}^{h\top }$ 

where the first equation represents truncation and second denotes absoprtion. The PCA basis $\tilde {\mathbf {Q}}_{v}^{g(h)}\in \mathbb {R}^{_{h}\times r}$  is truncated to rank r. As a re-sult, the output layer shares G PCA basis, reduc-ing the total computation required for PCA. Af-ter truncation, the shared value projection matrix and the per-head output projection matrix become $\tilde {\mathbf {W}}_{v}^{g(h)}\in \mathbb {R}^{r\times _{\text {hi}}}$ and $\tilde {\mathbf {W}}_{o}^{h}\in \mathbb {R}^{d_{\text {hid}}xr}$ ,respectively.

# B Truncation Loss Analysis

In the following, we provide the detailed proof on the direct mapping between eigenvalues and truncation loss in single-head and multi-head cases. **Theorem** **4.1** [Reconstruction Error of Single-head Output PCA Projection]LetYvh=XWvh\top,and $\tilde {\mathbf {Y}}_{v}^{h}=\mathbf {Y}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }$  be the rank-r approximation obtained by projectingYvhonto its top-r principal components. Then the squared Frobenius norm of the reconstruction error satisfies:

$$\left\|\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right\|_{F}^{2}=\sum _{i=r+1}^{d_{h}}λ_{i}^{h}$$

where{λih}are the eigenvaluesofYvh\topYvh.

Proof. The projection $\tilde {\mathbf {Y}}_{v}^{h}=\mathbf {Y}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }$  mini-mizes the Frobenius norm among all rank-r ap-proximations. Let $\mathbf {P}=\tilde {\mathbf {Q}}_{v}^{h}\tilde {\mathbf {Q}}_{v}^{h\top }\in \mathbb {R}^{d_{h}xd_{h}}$  be the orthogonal projector onto the top r eigenspace. The squared Frobenius norm of the reconstruction error is:

$$\left\|\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right\|_{F}^{2}=\left\|\mathbf {Y}_{v}^{h}(\mathbf {I}-\mathbf {P})\right\|_{F}^{2}$$

$$=\text {Trace}\left[\mathbf {Y}_{v}^{h}(\mathbf {I}-\mathbf {P})(\mathbf {I}-\mathbf {P})^{\top }\mathbf {Y}_{v}^{h\top }\right]$$

$$=\text {Trace}\left[\mathbf {Y}_{v}^{h}(\mathbf {I}-\mathbf {P})\mathbf {Y}_{v}^{h\top }\right]$$

$$=\text {Trace}\left[(\mathbf {I}-\mathbf {P})\mathbf {Y}_{v}^{h\top }\mathbf {Y}_{v}^{h}\right]$$

SinceYvh\topYvh=QvhΛvhQvh\top,let $\hat {\mathbf {Q}}_{v}^{h}=\mathbf {Q}_{v}^{h}-$ $\tilde {\mathbf {Q}}_{v}^{h},$ ,we have $\mathbf {I}-\mathbf {P}=\hat {\mathbf {Q}_{v}^{h}}\hat {\mathbf {Q}_{v}^{h\top }}$ ,and thus:

$$\left\|\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right\|_{F}^{2}=\text {Trace}\left[\hat {\mathbf {Q}}_{v}^{h}\hat {\mathbf {Q}}_{v}^{h\top }\mathbf {Q}_{v}^{h}\mathbf {Λ}_{v}^{h}\mathbf {Q}_{v}^{h\top }\right]$$

$$=\text {Trace}\left[\hat {\mathbf {Q}}_{v}^{h\top }\mathbf {Q}_{v}^{h}\mathbf {Λ}_{v}^{h}\mathbf {Q}_{v}^{h\top }\hat {\mathbf {Q}}_{v}^{h}\right]$$

$$=\text {Trace}\left[\hat {\boldsymbol {Λ}}_{v}^{h}\right]=\sum _{i=r+1}^{d_{h}}λ_{i}^{h}.$$

☐

**Corollary** **4.2** [Reconstruction Error of Multi-head Output PCA Projection] Let Yv=

Table 3: Comparison of zero-shot downstream performance wwith prior importance-based compression methods for Llama-2 7B at 20% compression ratio and Llama-2 13B at 30% compression ratio.

<table border="1" ><tr>
<td>Model</td>
<td>Method</td>
<td>Ratio</td>
<td>WinoG.</td>
<td> HellaS.</td>
<td>ARC-e</td>
<td>ARC-c</td>
<td>PIQA</td>
<td>Avg.</td>
<td>delta</td>
</tr><tr>
<td rowspan="5">LLaMA-2 7B</td>
<td>Original</td>
<td>0%</td>
<td>69.06</td>
<td>75.99</td>
<td>74.58</td>
<td>46.25</td>
<td>77.91</td>
<td>68.76</td>
<td>0</td>
</tr><tr>
<td>LaCo</td>
<td></td>
<td>60.46</td>
<td>54.08</td>
<td>55.39</td>
<td>35.84</td>
<td>68.34</td>
<td>54.82</td>
<td>-13.94</td>
</tr><tr>
<td>ShortGPT</td>
<td rowspan="2">20%</td>
<td>65.90</td>
<td>62.63</td>
<td>56.06</td>
<td>36.09</td>
<td>70.24</td>
<td>58.18</td>
<td>-10.58</td>
</tr><tr>
<td>BlockPruner</td>
<td>62.43</td>
<td>65.87</td>
<td>61.07</td>
<td>37.29</td>
<td>74.21</td>
<td>60.17</td>
<td>-8.59</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>20%</td>
<td>67.88</td>
<td>69.24</td>
<td>70.45</td>
<td>41.38</td>
<td>75.35</td>
<td>64.86</td>
<td>-3.90</td>
</tr><tr>
<td rowspan="5">LLaMA-2 13B</td>
<td>Original</td>
<td>0%</td>
<td>72.22</td>
<td>79.39</td>
<td>79.42</td>
<td>49.06</td>
<td>80.52</td>
<td>72.12</td>
<td>0</td>
</tr><tr>
<td>LaCo</td>
<td></td>
<td>59.27</td>
<td>60.44</td>
<td>54.34</td>
<td>34.56</td>
<td>72.42</td>
<td>55.44</td>
<td>-16.68</td>
</tr><tr>
<td>ShortGPT</td>
<td rowspan="2">25%</td>
<td>70.80</td>
<td>67.80</td>
<td>60.35</td>
<td>41.30</td>
<td>72.74</td>
<td>62.60</td>
<td>-9.52</td>
</tr><tr>
<td>BlockPruner</td>
<td>66.30</td>
<td>72.20</td>
<td>65.82</td>
<td>41.38</td>
<td>76.93</td>
<td>64.53</td>
<td>-7.59</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>25%</td>
<td>71.35</td>
<td>72.93</td>
<td>73.48</td>
<td>43.94</td>
<td>76.44</td>
<td>67.63</td>
<td>-4.49</td>
</tr></table>

<!-- FLAT-LLM (Ours) Slicegpt SVD-LLM Original Llama-2 7B Llama-2 13B Mistral 7B 65 70 60 65 60 (%) 55 (%) 60 20y 20 55 50 Acc(%) 55 6N 6N 50 6Y 50 45 45 45 40 40 40 10% 20% 30% 40% 10% 20% 30% 40% 10% 20% 30% 40% Compression Ratio Compression Ratio Compression Ratio -->
![](./images/4f54f6670c97e459.jpg)

Figure 8: Comparison of zero-shot average accuracy versus compression ratio on various models.

concat(Yv1,⋯YvH)be the concatenated output hid-den states. The squared Frobenius norm of the reconstruction error satisfies:

$$\left\|\mathbf {Y}_{v}-\tilde {\mathbf {Y}}_{v}\right\|_{F}^{2}=\sum _{h=1}^{H}\sum _{i=r+1}^{d_{h}}λ_{i}^{h}$$

Proof.

$$\left\|\mathbf {Y}_{v}-\tilde {\mathbf {Y}}_{v}\right\|_{F}^{2}=\left\|\text {concat}\left(\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right)\right\|_{F}^{2}\quad =\sum _{h=1}^{H}\left\|\left(\mathbf {Y}_{v}^{h}-\tilde {\mathbf {Y}}_{v}^{h}\right)\right\|_{F}^{2}=\sum _{h=1}^{H}\sum _{i=r+1}^{d_{h}}λ_{i}^{h}$$

<!-- ☐ -->
![](./images/54185b08b1439a39.jpg)

Therefore, the reconstruction loss of multi-head value output equals to the sum of the dropped eigen-values of all heads. Truncating the smallestdh-r eigenvalues of each head leads to the lowest recon-struction loss.

# C Nyström-based Low-rank MLP Approximation

To compress the MLP layers, we apply a structured Nyström approximation guided by data-dependent ridge leverage scores. Letdhidanddintdenote the hidden and intermediate dimensions, respectively. The method operates on the up-projection matrix W1∈Rhixitand the down-projection matrix W2∈Rit×hiwithin each MLLP block. To capture activation structure, we compute the cor-relation matrixCσover intermediate hidden states passed through the SiLU activation. Ridge lever-age scores are then derived from the regularized correlation matrix viaCσ(Cσ+I)-1,quantifying the relative importance of each intermediate chan-nel. Based on these scores, we construct a selection matrixSk∈Ri×k that retains the top-k most informative channels, where k is determined by the target sparsity ratio. The up-projection is com-pressed by selecting the top k columns,i.e.W1Sk, while the down-projection is approximated via Nyström reconstruction: (Sk\topCσSk)-1Sk\topCσW2. This data-aware procedure preserves key activation subspaces while significantly reducing parameter count and computation.

<!-- 100 100 80 80 (3)oHeNYUEN BuluIeueN 60 60 40 40 20 20 (8)oHea 6uuIeuaa ·62 0 0 0 5 10 15 20 25 30 Layer ID -->
![](./images/905822e8e0d1bbc0.jpg)

(a) Llama-3 8B

<!-- 100 100 80 80 (3)oe Nue BuIuIeue 60 60 40 40 20 20 (8)OHeN SuIuIeueN 6N 0 0 5 10 15 20 25 30 Layer ID -->
![](./images/29b3f76761e933a7.jpg)

(b) Mistral-7B

<!-- 100 100 80 80 (3)oHEZYUEN 5uluIEueN 60 60 40 40 20 20 (8)oIea 6uuIeue ·62 0 0 10 20 30 40 Layer ID -->
![](./images/5501917a289fdda4.jpg)

(c) Llama-2 13B

<!-- 100 100 80 80 (3)ole Nue BuIuIeuΦ 60 60 40 40 20 20 (8)olea BuIuIeue 6N 0 0 0 20 40 60 80 Layer ID -->
![](./images/82e0d806a05ed9e4.jpg)

(d) Llama-2 70B

Figure 9: Layer-wise sparsity score visualization across different LLM models using our IPRS algorithm.

Table 4: Comparison of calibration time and perfor-mance with other methods on LLAMA-2 13B at 20% compression ratio.

<table border="1" ><tr>
<td>Method</td>
<td>Time↓</td>
<td>PPL↓</td>
<td>Avg.ACC↑</td>
</tr><tr>
<td>SliceGPT</td>
<td>0h35m</td>
<td>7.10</td>
<td>0.51</td>
</tr><tr>
<td>SVD-LLM</td>
<td>0h27m</td>
<td>7.69</td>
<td>0.56</td>
</tr><tr>
<td>FLAP</td>
<td>0h10m</td>
<td>5.90</td>
<td>0.57</td>
</tr><tr>
<td>FLAT-LLM</td>
<td>0h15m</td>
<td>5.55</td>
<td>0.63</td>
</tr></table>

# D Additional Experimental Results

## D.1 Calibration Efficiency

Table 4 compares the calibration time, perplexity (PPL),and average zero-shot accuracy of various compression methods at a 20% compression ratio on Llama-2 13B. The results are collect from a single A100 GPU.Among the evaluated methods, FLAT-LLM achieves the best overall performance, attaining the lowest perplexity and the highest aver-age accuracy, while maintaining a moderate calibra-tion time of 15 minutes. In contrast, SliceGPT and SVD-LLM exhibit 11.8-2.3×10onger calibration times and significantly7-12%accuracy drop. Al-though FLAP achieves theshortest calibration time, it suffers from a 6% accuracy gap compared to our FLAT-LLM. These results highlight that FLAT-**LLM** offers the best trade-off between calibration efficiency and compression performance,demon-

<!-- 65 Original IPRS (Alpaca) (%) 60 Uniform (Alpaca) IPRS (WikiText-2) 22y Uniform (WikiText-2) 55 6N 50 45 10% 20% 30% 40% Compression Ratio -->
![](./images/e16bf91b0bc29d2d.jpg)

Figure 10: Comparison of average zero-shot accuracy across eight downstream datasets versus compression ratio using uniform rank and our IPRS algorithm on SVD-LLM.

strating high practical deployability.

## D.2 Additional Evaluation on Zero-shot Downstream tasks

Figure 8 presents a comprehensive comparison of average zero-shot accuracy across compression ra-tios ranging from 10% to 40%, evaluated on three LLM models: LLaMA-2 7B, LLaMA-2 13B,and Mistral-7B. Calibration is performed using 128samples with a sequence length of 4096 from the Alpaca dataset. The performance of the uncom-pressed dense model is shown as the upper bound (purple dashed line). Our proposed method,FLAT-LLM, consistently outperforms prior low-rank de-composition approaches, including SlicedGPT and SVD-LLM, across all models and compression levels. Notably, FLAT-LLM maintains high per-formance with less than a 2% accuracy drop at 10% compression across all models. These re-sults underscore the effectiveness and scalability of FLAT-LLM across varying models and compres-sion regimes.

We further evaluate FLAT-LLM against addi-tional importance-based baselines on five reason-ing tasks, as shown in Table 3. Following the setting of BlockPurner, we use 256 samples on Alpaca datasets for calibration. On average, FLAT-LLM incurs less than a 4% accuracy drop when pruning Llama-2 7B at a 20% compression ratio, and approximately a 6% drop on Llama-2 13B at a 30% compression ratio-significantly outper-forming all baselines. Moreover, FLAT-LLM con-sistently achieves the highest accuracy across all benchmarks, demonstrating strong generalization across tasks.

## D.3 Illustration for IPRS Rank Distribution

Figure 9 presents the layer-wise remaining rank ratios produced by the IPRS algorithm across vari-ous LLM models, including Llama-3 8B, Llama-213B,70B,and Mistral-7B. For each model, curves are shown under different target average remaining ratios, ranging from 30% to 90%. To compute the importance of each model, we use 128 samples with 4096 sequence length randomly selected from WikiText-2 dataset. The gray dashed line repre-sents the normalized importance scores t used to guide the adaptive rank allocation. Across all mod-els, IPRS consistently preserves more capacity in layers deemed more important, while aggressively pruning less critical layers. Notably, the resulting rank distributions exhibit a U-shaped or skewed U-shaped pattern, allocating higher ranks to early, middle, or final layers depending on model-specific importance trends. Despite variations in architec-ture and depth, all models share a common pattern in which the input-adjacent and output-adjacent layers are more heavily preserved, reflecting their broader importance for information transformation and representation. This consistent behavior across diverse models highlights the robustness and gener-alization ability of the IPRS algorithm in learning effective, non-uniform rank allocations tailored1 to model-specific importance profiles.

## D.4 IPRS on Other Decomposition Methods

To evaluate the generality of our rank selection method, we also apply IPRS to SVD-LLM, as shown in Figure 10. While it leads to improve-ments in this setting, the average accuracy gain is modest, typically up to 2.5%. This is due to the distinct truncation loss patterns across layers: SVD-LLM exhibits relatively uniform truncation loss,whereas FLAT-LLM displays highly variable loss, amplifying the benefit of adaptive rank alloca-tion. Overall, combining IPRS with our head-wise PCA-based compression in FLAT-LLM yields con-sistently supperior performance, underscoring the complementary strengths of fine-grained PCA and importance-aware rank selection.

