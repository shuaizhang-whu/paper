# Activation-Informed Pareto-Guided Low-Rank Compression for Efficient

# LLM/VLM

**Ryan Solgi¹, Parsa Madinei¹, Jiayi Tian¹, Rupak Swaminathan²,**

**Jing Liu², Nathan Susanj²,Zheng Zhang¹**

¹University of California-Santa Barbara, USA

2Amazon, USA

solgi@ucsb.edu,zhengzhang@ece.ucsb.edu

# Abstract

Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose sig-nificant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this chal-lenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimiza-tion and prove that a single uniform toler-ance yields surrogate Pareto-optimal heteroge-neous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value De-composition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternat-ing least-squares implementation. We apply PGSVD to both LLM and VLM, showing bet-ter accuracy at the same compression levels and inference speedup.

# 1 Introduction

Pre-trained foundation models have achieved state-of-the-art performance in diverse do-mains (Vaswani et al., 2017; Bommasani et al., 2021). However, their huge memory and comput-ing demands pose significant barriers to efficient deployment on various platforms (Patterson et al., 2021; Strubell et al., 2019). Hence, compressing these models has become an active area of research. Different methods have been studied, including pruning (Sanh et al., 2020; Someki et al., 2025), distillation (Sanh et al., 2019), quantization (Shen et al., 2020), and low-rank factorization (Lan et al., 2020; Hsu et al., 2022a; Gao et al., 2024).

In this work, we investigate zero-shot compres-sion of pre-trained models through low-rank factor-ization. Zero-shot compression has gained popular-ity due to its ability to rapidly reduce memory and computation requirements while preserving down-stream performance without re-training (Frantar et al., 2022; Dettmers et al., 2023; Frantar et al., 2023). In addition to parameter and memory reduc-tion,low-rank compression can also yield real-time inference speedup on standard platforms. How-ever, some (or even many) layers in a pre-trained model may not exhibit a low-rank property, lead-ing to dramatic accuracy drop in direct factoriza-tion. To address this, several studies leverage gra-dients or activations to guide factorization (Hsu et al., 2022b; Yuan et al., 2023). Among activation-based compression methods, SVD-LLM(Wang et al., 2024, 2025) factorizes LLM layers using truncation-sensitive data whitening techniques. De-spite promising results, previous methods typi-cally apply the same compression ratio across all layers (Wang et al., 2024), or apply heuristic rank selection methods with no theoretical guar-antee (Wang et al., 2025; Zhang et al., 2024). Given the large number of layers in LLMs and the cross-modality disparities in multimodal (e.g., vision-language) models (Yang et al.,2024),choos-ing adaptive compression ratios between layers remains a challenge. Furthermore, without suf-ficient theoretical insights, some layers may be over-compressed or under-compressed, leading to overall performance degradation.

This paper tries to address two key challenges: (i) the absence of a theory that links layer-wvise compression to overall model performance, and(ii) the reliance on uniform or heuristic compression ratios. To overcome these issues, we first reveal how the errors of activation-based layer compres-sion propagate to and impact the overall network loss. Furthermore, from a bi-objective perspective, we formally demonstrate that every uniform error-tolerance allocation is surrogate Pareto-optimal and automatically induces heterogeneous ranks across layers,collapsing a difficult search over many ranks to a single knob as illustrated in Fig. 1. These

<!-- 1 -->

<!-- S7071o0LITO.so] IAttSS0.0ISC:AIXT -->

<!-- Unimodal 3 Text Compression Ratio Error Tolerance -->
![](./images/776dce1ab20cb206.jpg)

<!-- Multimodal ϵt ϵv Text Image -->
![](./images/aae12380a5b13d14.jpg)

Figure 1: Overview of PGSVD: (left) unimodal model using a uniform error tolerance that yields heteroge-neous compression ratios; (right) multimodal model with separate uniform tolerances for each tower.

theoretical insights motivate us to propose Pareto-Guided Singular Value Decomposition (PGSVD) to improve the performance of low-rank LLM/VLM compression. Our main contributions can be sum-marized as follows:

**·Theoretical Insights about SVD-Based LLM** **Compression.** We formulate compression-ratio allocation across layers as a bi-objective op-timization problem. By linking network loss with layerwise activation-based compression, we rigorously show that a uniform error-tolerance allocation across layers yields a data-agnostic, surrogate-optimal architecture, defining a near Pareto-optimal trade-off between network loss and compression ratio.

**·New Compression Algorithms via Pareto-Guided Rank Selections.** We propose PGSVD, a novel zero-shot compression framework that in-tegrates Pareto-guided rank selection with SVD to jointly achieve network-level and layer-wise optimality. PGSVD is further equipped with an alternating least squares solver for efficiently up-dating low-rank factors based on activations.

**·Experimental Results on LLMs and VLMs.** We demonstrate that PGSVD outperforms prior activation-aware low-rank compression methods across multiple modeI sizes and datasets, and we further extend the approach to VLMs, high-lighting its applicability beyond unimodal archi-tectures. Our experiments show that PGSVD achieves more than a 30% accuracy imnprovement over uniform compression ratio assignments with the same memory and inference speedup.

# 2 Background

**Activation-Aware Compression.** In low-rank compression of LLMs, the weight matrix of a layer denoted byW∈RN×Mis replaced by its low-rank approximation $\hat {\mathbf {W}}=\mathbf {AB}$ whereA∈RNxr and B∈Rr×M. Traditionally, this compression is done by computing a best low-rank approximation:

$$\min _{\mathbf {A}\in \mathbb {R}^{Nxr},\mathbf {B}\in \mathbb {R}^{rxM}}\|\mathbf {W}-\mathbf {AB}\|_{F}^{2}\tag{1}$$

The optimal solution isA★=UrΣr1/2andB★= Σr1/2VrTwhereUr,,Σr andVrare the truncated SVD factors corresponding to the top-r singular values of W. However, the weights of a pre-trained LLM/VLM may not have low-rank proper-ties at some layers, which lead to dramatic perfor-mance loss in the above compression. In contrast, activations (X) have exhibited low-rank proper-ties (Zhang et al., 2024). Therefore, it is more promising to shift the main objective of the com-pression paradigm from Eq. (1) to the following:

$$\min _{\mathbf {A}\in \mathbb {R}^{Nxr},\mathbf {B}\in \mathbb {R}^{rxM}}\|\mathbf {WX}-\mathbf {ABX}\|_{F}^{2}\tag{2}$$

**SVD-LLM.** A reformulation of (2) based on whitened activations was proposed as follows:

$$\min _{\mathbf {A}\in \mathbb {R}^{Nxr},\mathbf {B}\in \mathbb {R}^{rxM}}\quad \left\|\mathbf {WX}-\mathbf {ABT}^{-1}\mathbf {X}\right\|_{F}^{2}\tag{3}$$

**where** T is Cholesky factor ofXX\top(Wang et al., 2024). The optimal solution is given by A★= $\overline {\mathbf {U}}_{r}\bar {Σ}_{r}^{1/2}\text {ad}\mathbf {B}^{\star }=\bar {Σ}_{r}^{1/2}\overline {\mathbf {V}}_{r}^{\top }$ ,where $\overline {\mathbf {U}}_{r}\bar {Σ}_{r}\overline {\mathbf {V}}r^{\top }$ is the rank-rSVD of WT.

# 3 Pareto-Guided Rank Selection

The performance of a compressed LLM/VLM highly depends on the choice of matrix rank per layer. For a foundation model with many lay-ers,this selection becomes particularly challenging. To address this challenge, this section formulates model compression as a bi-objective optimization problem which minimizes both the network loss and the total number of model parameters with re-spect to the layer-wise compression ratios. To solve the proposed bi-objective problem, we reveal the connection between the the network loss and the layer-wise approximation error, and employ a rigor-ous analytical approach to derive a surrogate Pareto frontier. Finally, we demonstrate that our solution guarantees the optimal allocation of compression ratios throughout the whole network in surrogate sense.

<!-- 2 -->

## 3.1 Bi-Objective Compression Formulation

Consider a weight matrixWl∈RNl×Mlin layer l and its rank-rl compressed weight $\hat {\mathbf {W}}_{l}^{\left(r_{l}\right)}$  We define the set of matrix ranks

$$Γ_{l}:=\left\{r_{l}\in \mathbb {Z}|e_{l}=\frac {\left\|\hat {\mathbf {W}}_{l}^{\left(r_{l}\right)}-\mathbf {W}_{l}\right\|_{F}}{\left\|\mathbf {W}_{l}\right\|_{F}}\leq \varepsilon _{l}\right\}$$

with 0≤ϵl≤1 (4)

Letr=[r1,r2,⋯,rL]specify the ranks of all L layers and|ΔL(r)|denote the absolute change in network loss due to replacing every weight ma-trixWlwith its approximation $\hat {\mathbf {W}}_{l}^{\left(r_{l}\right)}$ .The total number of parameters of the compressed LLM is

$$S(\boldsymbol {r})=\sum _{l=1}^{L}P_{l}\left(r_{l}\right)\tag{5}$$

wherePl(rl)is the total number of low-rank pa-rameters in layer l.

We aim to jointly minimize the number of pa-rameters and the loss change during compression:

<table border="1" ><tr>
<td>Formulation 1 (Bi-objective Compression). minr∈∏l=1LΓl(S(r),|ΔL(r)|) (B)</td>
</tr></table>

To derive a surrogate Pareto frontier of formula-tion 1 we first link|ΔL(r)|to layer-wise compres-sion errors in the following section.

## 3.2 Activation-Aware Compression and Network Loss

We show that the activation-based compression ob-jective Eq. (2) serves as an upper bound on the network loss change. Not only does this result highlight why activation-based compression is ef-fective, but it also connects the network loss to the layer-wise compression, which we will utilize to solve our optimization problem.

**Theorem** **1** (Loss Sensitivity to Activation-Based Compression).Letxl+=σ(Wlxl),with batch Xl=[\begin{array}lllxl(1)&⋯&xl(B)\end{array}],where σ acts elementwise and susupt∈R|σ′(t)|≤c&lt;∞and $\hat {\mathbf {W}}_{l}=\mathbf {W}_{l}+$ ΔWldenote the compressed weights. Then,for a differentiable scalar loss C we have:

$$|\Delta \mathcal {L}|\leq G\sum _{l=1}^{L}\left(\prod _{m=l+1}^{L}\mathcal {K}_{m}\right)c\left\|\Delta \mathbf {W}_{l}\mathbf {X}_{l}\right\|_{F}$$

whereG:=\|∇YL\|F,Y=XL+1,Kl:= sup1≤i≤B\|Jl(i)\|,Jl(i)=diag(σ′(Wlxl(i)))Wl.

Proof. The proof follows from a first-order pertur-bation analysis (Appendix A). ☐

Theorem 1 shows how the weight changes caused by compression affect the overall loss. The local effect of the layer l is measured by how strongly the perturbed weights distort its activa-tions,quantified by\|ΔWlXl\|F. This error is then scaled by the activation slope C and by the product of Jacobian norms oof all subsequent layers ∏m=+1LKmwhich captures how much the sub-sequent transformations can amplify or dampen the perturbation. Finally, the loss gradient G con-verts this distortion into an upper bound on the loss change. Theorem 1 establishes a formal re-lation between network loss and the activation-aware compression objective, \|ΔWlXl\|F,for each layer l. For a fixed pre-trained model and dataset, the coefficients are constants, so mini-mizing∑l=1L\|ΔWlXl\|Feffectively minimizes a provable first-order surrogate bound on the loss change, although there is still slack from norm in-equalities and higher-order terms.

## 3.3 Surrogate Pareto Frontier

This subsection first proposes a scalarized surro-gate, referred to as the rank allocation problem. Then we show that this problem is equivalent to a layer-wise error-tolerance allocation problem. Us-ing this equivalence, we show that every uniform error-tolerance allocation across layers defines a point on the surrogate Pareto frontier of our bi-objective optimization. Therefore, our formulation guarantees optimal compression ratio allocations throughout the network in the surrogate sense.

We define

$$α_{l}=\left\|\nabla _{\mathbf {Y}}\mathcal {L}\right\|_{F}\left(\prod _{m=l+1}^{L}\mathcal {K}_{m}\right)c\left\|\mathbf {X}_{l}\right\|_{F}\left\|\mathbf {W}_{l}\right\|_{F}$$

Letb be a budget for the total number of parameters of the compressed model. Using Theorem 1 and inequality\|ΔWX\|F≤\|ΔW\|F\|X\|Fwe have the scalarized surrogate of Formulation 1.

<!-- 3 -->

<table border="1" ><tr>
<td>Formulation 2 (Rank Allocation).<br>minr∈∏l=1LΓl|ΔL(r)|≤∑l=1Lαlel(rl)<br>•<br>.<br>∑l=1LPl(rl)≤b (P)</td>
</tr></table>

In the following, we define a mapping from rank (compression ratio) to compression error tolerance and formally establish our error-equivalence for-mulation in Proposition 1.

**Definition 1**(ε-Parameter Mapping via SVD).For any matrixW∈RN×Mand tolerance ϵ∈[0,1]],there exists a unique minimal rankr★(ϵ)such that the truncated SVD approximation $\hat {\mathbf {W}}^{\left(r^{\star }\right)}$ satisfies

$$\frac {\left\|\mathbf {W}-\hat {\mathbf {W}}^{\left(r^{\star }\right)}\right\|_{F}}{\|\mathbf {W}\|_{F}}\leq \varepsilon .$$

Becauser★(ϵ)is minimal, it induces the minimal number of parameters required to achieve error at most3. We define the ϵ parameter mapping

$$h:[0,1]\rightarrow \mathbb {Z}_{\geq 0}\quad h(\varepsilon ):=P\left(r^{\star }(\varepsilon )\right)$$

whereP(r)=r(M+N)denotes the number of parameters of a rank-r SVD factorization.

**Proposition** **1** (Rank-ϵ Allocation Equivalence). The rank-allocation problem (P) and the 3-allocation problem (E) have the same optimal value.

Proof. See Appendix B.

<table border="1" ><tr>
<td colspan="2">Formulation 3(ε-Allocation)</td>
</tr><tr>
<td>min0≤ϵ1,⋯,ϵL≤1∑l=1Lαlϵl s.t.∑l=1Lhl(ϵl)≤b</td>
<td>(E)</td>
</tr></table>

Although rank allocation can be posed as a knap-sack problem (Formulation 2), the large number of layers and candidate ranks typically makes this approach computationally intractable. Formula-tion 3 proovides a continuous surrogate that is more computationally efficient, convex, and admits a closed-form solution, under appropriate conditions. Rank selection can thus follow two approaches. A data-driven method requires the coefficients αhowever, the activations **X** are unknown, sampling only yields approximations, and the resulting low-rank architecture becomes data-dependent,which can be undesirable in deployment. In contrast, a data-agnostic method assumes equal weighting across layers, motivated by robust optimization (Appendix D); while conservative, this strategy avoids data dependence, improves robustness to uncertainty, and ensures that changes in the data do not require altering the network architecture, but only updating the low-rank parameters. We therefore adopt the latter method and show that assigning the same error tolerance (ε) to all layers yields heterogeneous ranks that are optimal, in the surrogate sense, for the bi-objective Formulation 1, as formally stated in Theorem 2.

**Lemma** **1** (Uniform 3 under homogeneous sen-sitivity and bounded profiles). Consider the3allocation problem (E) for a homogeneous network (where.αl=α,∀l)to ensure robustness followed by Remark 1 (see Appendix D) and denote by

p=min0≤ϵ1,⋯,ϵL≤1∑l=1Lαϵl s.t.∑l=1Lhl(ϵl)≤b.

Assume each layer's parameter function hlis bounded by common nonincreasing convex en-velopes $\underline {h},\bar {h}:[0,1]\rightarrow \mathbb {R}_{\geq 0}$ (see Appendix G) such that

$$\underline {h}(\varepsilon )\leq h_{l}(\varepsilon )\leq \bar {h}(\varepsilon )\quad \forall l,\varepsilon \in [0,1].$$

Let the budgetb satisfy $L\bar {h}(1)\leq \leq L\bar {h}(0)$ .Then the 3-allocation problem admits a uniform solution ϵ1=⋯=ϵLthat (i) is optimal for the symmetric surrogate using $\bar {h}$   with the common valueϵ★satis-fying $L\bar {h}\left(\varepsilon ^{\star }\right)=b$ (ii)is minimax-optimal for the worst-case admissible profiles, attained at $h_{}=\bar {h}$ and (iii))yields bounds

$$αL\varepsilon ^{\ell }\leq p\leq αL\varepsilon ^{\mathrm {u}}$$

whereϵ\ell,ϵuare defined by $L\underline {h}\left(\varepsilon ^{\ell }\right)=$ $,L\bar {h}\left(\varepsilon ^{\mathrm {u}}\right)=$ b.

Proof. See Appendix C.

☐

**Theorem** **2** (Uniform 3 yields the surrogate Pareto front of (B)). Under the homogeneous sensitivity and bounded-profile assumption (Lemma 1),every uniform toleranc ϵ∈[0,1]corresponds to a point on the surrogate Pareto frontier of(B) , which is the set of nondominated pairs

$$\left(\sum _{l=1}^{L}α_{l}e_{l}\left(r_{l}\right),\sum _{l=1}^{L}P_{l}\left(r_{l}\right)\right)$$

<!-- 4 -->

Proof. See Appendix E.

☐

Theorem 2 shows that every uniform layer-wise error tolerance (ε) for SVD compression yields a near Pareto-optimal solution for Formulation 1. Since layer spectra differ, the same 3 naturally provides **adaptive rank selections** across layers. Whereas previous work often applied uniform com-pression ratios across layers (Wang et al., 2024; Yuan et al., 2023), we suggest instead applying a uniform error allocation directly tied to netwvork loss. This approach leaves a single hyperparameter 3 to control the loss-compression tradle-off.When finer control is needed, Theorem 1 applies to clus-tered allocations, where layers are grouped into classes (e.g., **MLP** and attention) and a common εis assigned within each class.

# 4 From Theory to Algorithms

**PGSVD.** Based on the theoretical results in Sec-tion 3, we propose the PGSVD to improve the LLM/VLM performance during compression via activation-aware SVD. PGSVD first determines the ranks and initializes the factors by directly factorizing W. It then further optimizes A and **B** to minimize Eq. (2). Following Theorem 2, PGSVD assigns a uniform compression tolerance 3 to all layers, which naturally leads to heteroge-neous compression ratios. After determining the optimal layer-wise ranks through this allocation, PGSVD minimizes Eq. (2) with respect to the low-rank parameters using activations.

**Efficient ALS Implementation.** We further in-troduce a new alternating least squares (ALS) solver to improve the efficiency of the compres-sion algorithm. By expanding Eq. (2) in trace form and differentiating with respect to A and B(see Appendix F), we obtain the following ALS updates:

$$\mathbf {A}=\mathbf {WMB}^{\top }\left(\mathbf {BMB}^{\top }\right)^{\dagger },\quad \mathbf {B}=\left(\mathbf {A}^{\top }\mathbf {A}\right)^{\dagger }\mathbf {A}^{\top }\mathbf {W}\tag{6}$$

whereM=XXTis the empirical covariance ma-trix and A and **B** are initialized by the rank-r ap-proximation of **W** via SVD. After initialization,A and B are updated for T iterations. Because the pseudo inverses involve onlyrxmatrices, these steps are computationally efficient. The algorithm 1summarizes the PGSVD steps.

**Algorithm 1 PGSVD Algorithm**

**Require:** {Wl}l=1L,{Ml}l=1L,ϵ,

**for**l=1**to** L **do**

$$r\leftarrow \min \left\{r\in \mathbb {Z}|e_{l}(r)\leq \varepsilon \right\}$$

$$\mathbf {U}_{r}Σ_{r}\mathbf {V}_{r}^{\top }\leftarrow \text {SVD}\left(\mathbf {W}_{l},r\right)$$

InitializeAl=UrΣr1/2andBl=Σr1/2Vr\top

**for**t=1toτdo

UpdateAlandBlvia ALS [Eq.(6)]

**end for**

**end for**

**Return**{Al,Bl}l=1L

**PGSVD for Multimodal Models.** VLM with modality-specific towers (e.g., a ViT image encoder and a Transformer text encoder) exhibit different weight and gradient distributions. This induces inter-modality imbalance, making a single global compression setting systematically biased. To ad-dress this, we assign separate error tolerancesϵv andϵtfor the vision and text modalities, respec-tively.Concretely, PGSVD applies a uniform toler-ance within each modality to yield heterogeneous, layer-wise ranks tailored to that modality's spec-trum, then optimizes the factors via activation least squares. This two-hyperparameter design preserves a small search space while respecting modality asymmetry, producing better accuracy-efficiency trade-offs than a single global3.

# 5 Experiments

We evaluated the performance of our proposed method on both LLM and VLM benchmarks. We also compared our method with previous activation-based low-rank compression and pruning base-lines. Since our method is most relevant to SVD-LLM (Wang et al., 2024), we first focus on compar-ison with SVD-LLM in Section 5.1. Furthermore, to isolate the effect of heterogeneous rank selection, we implemented a variant of SVD-LLM by replac-ing the rank selection step in the PGSVD algorithm with a set of prefix ranks determined by a uniform compression ratio across all layers. We refer to this method as SVD-ALS.

We implemented our approach using the Hug-ging Face Transformers library and compressed all linear layers in the self-attention modules and all linear projections in the MLP blocks. To ensure nu-merical stability, we used full precision for model inferences when computing covariance matrices and we used double precision during compression.

<!-- 5 -->

Table 1: Perplexity (PPL) and zero-shot accuracy (%) for reasoning tasks at Base, 20% and 40% compression.

<table border="1" ><tr>
<td>Model</td>
<td>Compression</td>
<td>Method</td>
<td>PPL(↓)</td>
<td>ARC-E </td>
<td>CSQA</td>
<td>Lambada</td>
<td> PIQA</td>
<td> Wino</td>
<td>AvgAcc(↑)</td>
</tr><tr>
<td rowspan="7">LLaMA-2-7B</td>
<td></td>
<td>Base</td>
<td>5.11</td>
<td>76.30</td>
<td>33.01</td>
<td>68.25</td>
<td>78.07</td>
<td>69.06</td>
<td>64.94</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD-LLM</td>
<td>7.70</td>
<td>68.56</td>
<td>19.82</td>
<td>46.61</td>
<td>70.35</td>
<td>64.33</td>
<td>53.93</td>
</tr><tr>
<td>SVD-ALS</td>
<td>7.72</td>
<td>68.73</td>
<td>20.80</td>
<td>47.18</td>
<td>70.57</td>
<td>64.80</td>
<td>54.42</td>
</tr><tr>
<td>PGSVD</td>
<td>7.38</td>
<td>70.75</td>
<td>20.80</td>
<td>53.02</td>
<td>71.27</td>
<td>64.56</td>
<td>56.08</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD-LLM</td>
<td>14.95</td>
<td>50.21</td>
<td>19.49</td>
<td>16.17</td>
<td>60.94</td>
<td>58.64</td>
<td>41.09</td>
</tr><tr>
<td>SVD-ALS</td>
<td>15.03</td>
<td>50.88</td>
<td>19.16</td>
<td>16.59</td>
<td>61.21</td>
<td>58.88</td>
<td>41.35</td>
</tr><tr>
<td>PGSVD</td>
<td>13.46</td>
<td>54.76</td>
<td>19.25</td>
<td>21.66</td>
<td>62.89</td>
<td>59.75</td>
<td>43.66</td>
</tr><tr>
<td rowspan="7">LLaMA-2-13B</td>
<td></td>
<td>Base</td>
<td>4.57</td>
<td>79.46</td>
<td>46.60</td>
<td>70.33</td>
<td>79.11</td>
<td>72.30</td>
<td>69.56</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD-LLM</td>
<td>6.17</td>
<td>71.00</td>
<td>25.23</td>
<td>57.54</td>
<td>72.91</td>
<td>67.17</td>
<td>58.77</td>
</tr><tr>
<td>SVD-ALS</td>
<td>6.19</td>
<td>66.12</td>
<td>20.31</td>
<td>45.97</td>
<td>69.26</td>
<td>63.46</td>
<td>53.02</td>
</tr><tr>
<td>PGSVD</td>
<td>5.96</td>
<td>73.36</td>
<td>25.14</td>
<td>60.78</td>
<td>73.23</td>
<td>69.38</td>
<td>60.38</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD-LLM</td>
<td>10.00</td>
<td>56.61</td>
<td>19.49</td>
<td>26.47</td>
<td>63.22</td>
<td>60.85</td>
<td>45.33</td>
</tr><tr>
<td>SVD-ALS</td>
<td>10.06</td>
<td>57.87</td>
<td>22.11</td>
<td>26.86</td>
<td>63.60</td>
<td>62.12</td>
<td>46.51</td>
</tr><tr>
<td>PGSVD</td>
<td>9.55</td>
<td>59.34</td>
<td>19.98</td>
<td>31.71</td>
<td>64.15</td>
<td>62.67</td>
<td>47.57</td>
</tr><tr>
<td rowspan="7">Mistral-7B</td>
<td></td>
<td>Base</td>
<td>4.92</td>
<td>80.85</td>
<td>56.27</td>
<td>69.49</td>
<td>80.63</td>
<td>74.19</td>
<td>72.29</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD-LLM</td>
<td>7.06</td>
<td>69.87</td>
<td>22.03</td>
<td>50.32</td>
<td>71.60</td>
<td>65.19</td>
<td>55.80</td>
</tr><tr>
<td>SVD-ALS</td>
<td>7.10</td>
<td>70.88</td>
<td>21.95</td>
<td>50.38</td>
<td>71.65</td>
<td>65.82</td>
<td>56.14</td>
</tr><tr>
<td>PGSVD</td>
<td>6.71</td>
<td>72.31</td>
<td>21.05</td>
<td>52.80</td>
<td>73.07</td>
<td>66.46</td>
<td>57.14</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD-LLM</td>
<td>16.30</td>
<td>46.30</td>
<td>19.41</td>
<td>15.93</td>
<td>59.25</td>
<td>57.14</td>
<td>39.61</td>
</tr><tr>
<td>SVD-ALS</td>
<td>15.69</td>
<td>47.18</td>
<td>19.49</td>
<td>16.13</td>
<td>58.76</td>
<td>57.62</td>
<td>39.84</td>
</tr><tr>
<td>PGSVD</td>
<td>14.43</td>
<td>49.24</td>
<td>20.23</td>
<td>16.48</td>
<td>60.45</td>
<td>58.48</td>
<td>40.98</td>
</tr></table>

In all experiments, 10 iterations were used for ALS in both SVD-ALS and PGSVD.

## 5.1 LLM Compression Results

**Experimental Setup.** We first evaluate PGSVD against the most relevant low-rank baselines, SVD-LLM and SVD-ALS, to study the effect of Pareto-guided heterogeneous rank allocation on several LLMs, including LLaMA-2 7B and 13B (Tou-vron et al., 2023) and Mistral-7B (AI, 2023). Be-cause PGSVD extends activation-based low-rank compression, our comparison focuses on methods that share the same compression framework and objective, ensuring a consistent and fair evalu1a-tion. We report both perplexity on the WikiText-2dataset (Merity et al., 2016) and zero-shot accuracy on a range of downstream reasoning tasks.

**Reasoning** **Tasks.** For downstream evaluation, we use the LM Evaluation Harness and benchmark performance of all LLMs on ARC-Easy (Clark et al., 2018), CommonsenseQA (Talmor et al., 2019), PIQA (Bisk et al., 2020),RACE (Lai et al.,2017),Winogrande (Sakaguchi et al.,2020), and LAMBADA-Standard (Paperno et al.,2016) listed in Table 1. Table 5 reported additional results on other reasoning tasks including ARC-Challenge (Clark et al., 2018), BoolQ (Clark et al., 2019),MMLU (Hendrycks et al., 2021), Hel-laSwag (Zellers et al.,2019),MathQA (Zellers

<!-- 50 40 (SanuIA)auL UoIssalduo0 30 20 10 0 LLaMA-2-7b LLaMA-2-13b Mistral-7b ■ALS (20%) Cholesky(20%) ■EVD (20%) ■ALS (40%) Cholesky(40%) EVD(40%) -->
![](./images/1c1d57641260e3da.jpg)

<!-- 8 AaIxeldJad 7.5 7 0 5 10 15 ALS Iterations -->
![](./images/980cb844d5e19c37.jpg)

Figure 2: Compression times of different solvers for different models (top) and perplexity versus the number of ALS iterations for LLaMA-2-7B (bottom).

et al., 2019), and Race (Lai et al., 2017).

**Comparison with Activation-Aware Compres-sion** **Methods** Table 1 shows that Pareto-guided rank selection substantially improves model per-formance in terms of both perplexity and accuracy on reasoning tasks, with gains of up to 30% and an average improvement of 14%. Whereas SVD-

<!-- 6 -->

<!-- 4000 3500 3000 S/SuexOL 2500 2000 1500 1000 256 1024 4096 Sequence Length ■Base SVD-ALS(20%) ■PGSVD(20%) ■SVD-ALS(40%) ■PGSVD (40%) -->
![](./images/217027084cb0f2a1.jpg)

<!-- 3500 3000 2500 S/suexOL 2000 1500 1000 256 1024 4096 Sequence Length ■Base ■SVD-ALS (20%) ■PGSVD (20%) ■SVD-ALS(40%) ■PGSVD (20%) -->
![](./images/16b7c78022eb0dea.jpg)

Figure 3: Inference throughput of LLaMA-2-7b (left) and Mistral 7b (right) for 20% and 40% compression using PGSVD and SVD-ALS compared to the base model.

ALS enforces a uniform compression ratio across all layers, PGSVD allocates Pareto-guided com-pression ratios, leading to a more balanced and effective utilization of model capacity. In addition, PGSVD outperforms SVD-LLM, achieving gains of up to 33% on reasoning tasks and more than 6% on average. Also, the ALS solver eliminates numerical failures common in Cholesky decom-position of SVD-LLM and yields faster compres-sion than eigenvalue decomposition (EVD)(Wang et al., 2025), as shown in Fig. 2. We further studied the effect of the number of ALS iterations on the quality of compression in PGSVD. The results for LLaMA-2 7B are shown in Fig. 2. We observed that improvements in perplexity plateauafter ap-proximately five to ten iterations. Notably,even with a single ALS iteration, the perplexity drops to an acceptable and competitive range.

**Inference Throughput.** PGSVD improves the inference throughput of the network compared to the uncompressed (base) models (Fig. 3) across dif-ferent compression ratios and sequence lengths for both LLaMA-2 and Mistral models on a H100 GPU. Since PGSVD assigns heterogeneous compression ratios, we also compared its inference throughput with SVD-ALS, which assigns homogeneous com-pression ratios. Under a naïve Python implementa-tion,PGSVD achieves almost the same inference speedup as SVD-ALS.

## 5.2 VLM Compression Results

**Experimental** **Setup.** We further evaluate the performance of PGSVD for zero-shot compres-sion of the CLIP model (Radford et al., 2021) on several standard benchmarks, including Cal-tech101 (Fei-Fei et al., 2004), Food101 (Bossard et al., 2014), OxfordPets (Parkhi et al., 2012), Stan-fordCars (Krause et al., 2013), EuroSAT (Helber et al., 2019), and DTD (Cimpoi et al., 2014). In this experiment, we include traditional SVD, activation-aware SVD (SVD-ALS), and our Pareto-guided variant (PGSVD) to examine both the effect of activation-based compression and the benefit of rank allocation. We use SVD rather than SVD-LLM as a baseline because activation-based com-pression has not previously been applied to vision-language models, and SVD provides a direct refer-ence for isolating the improvement introduced by activation-based compression.

**Background.** The successful compression of VLMs has remained challenging due to their ar-chitectural complexity and cross-modality dispar-ities (Ye et al.,2023;Li et al., 2023;Shi et al., 2023). Recently, ECoFLaP (Yang et al., 2024) achieved 40-60% unstructured sparsity in VLMs with minimal accuracy loss. However, because this sparsity is both relatively low and unstructured,it does not translate into proportional real-time mem-ory savings on standard platforms while low-rank compression can achieve actual memory reduction.

**Performance Evaluation.** Table 2 lists the Top-1 and Top-5 accuracies of the CLIP model com-pressed with SVD, SVD-ALS, and PGSVD,com-pared with the uncompressed (base) model. NNaïve SVD reduces accuracies in almost all benchmarks to near zero. SVD-ALS, significantly improves the accuracies since activation is considered in com-pression. PGSVD achieves thebestaccuracyacross all datasets, closing the gap with the base model. In most datasets, we observe greater sensitivity to the vision model than to the language model.

<!-- 7 -->

Table 2: Top-1 and Top-5 accuracies (Top-1/Top-5) for zero-shot compression of CLIP across six datasets.

<table border="1" ><tr>
<td>Compression</td>
<td>Method</td>
<td>Caltech101</td>
<td>Food101</td>
<td>OxfordPets</td>
<td>StanfordCars</td>
<td>Eurosat</td>
<td>DTD</td>
<td>Average</td>
</tr><tr>
<td></td>
<td>Base</td>
<td>86.39/99.02</td>
<td>88.51/98.61</td>
<td>89.94/97.52</td>
<td>64.22/93.86</td>
<td>39.00/86.03</td>
<td>44.25/74.95</td>
<td>68.72/91.67</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD</td>
<td>0.63/2.97</td>
<td>1.20/5.07</td>
<td>3.76/14.45</td>
<td>0.57/2.84</td>
<td>11.12/61.40</td>
<td>1.27/10.59</td>
<td>3.09/16.22</td>
</tr><tr>
<td>SVD-ALS</td>
<td>69.22/92.13</td>
<td>75.03/95.05</td>
<td>36.82/67.02</td>
<td>51.73/88.21</td>
<td>24.05/66.86</td>
<td>16.32/35.00</td>
<td>45.53/74.05</td>
</tr><tr>
<td>PGSVD</td>
<td>86.61/99.20</td>
<td>84.57/97.78</td>
<td>84.06/97.17</td>
<td>56.63/90.82</td>
<td>37.24/72.31</td>
<td>26.49/46.43</td>
<td>62.60/83.95</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD</td>
<td>0.63/3.29</td>
<td>0.90/5.00</td>
<td>2.92/13.30</td>
<td>0.67/2.38</td>
<td>11.10/52.56</td>
<td>3.46/10.79</td>
<td>3.28/14.55</td>
</tr><tr>
<td>SVD-ALS</td>
<td>73.35/94.53</td>
<td>69.81/92.53</td>
<td>7.52/26.62</td>
<td>31.94/71.88</td>
<td>20.69/69.37</td>
<td>19.15/40.69</td>
<td>37.08/65.94</td>
</tr><tr>
<td>PGSVD</td>
<td>76.95/95.67</td>
<td>72.48/93.33</td>
<td>55.68/83.02</td>
<td>39.77/78.70</td>
<td>38.95/69.57</td>
<td>21.49/36.80</td>
<td>50.89/76.18</td>
</tr></table>

Table 3: Comparison of PGSVD with other baselines on the WikiText-2 dataset.

<table border="1" ><tr>
<td rowspan="2">Model</td>
<td rowspan="2">Methods</td>
<td colspan="3">Compression</td>
</tr><tr>
<td>10%</td>
<td>30%</td>
<td>50%</td>
</tr><tr>
<td rowspan="4">LLaMA-2-7B<br>(PPL=5.11)</td>
<td>PGSVD</td>
<td>6.52</td>
<td>9.20</td>
<td>27.46</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>7.11</td>
<td>13.56</td>
<td>31.05</td>
</tr><tr>
<td>SliceGPT</td>
<td>6.69</td>
<td>11.94</td>
<td>25.84</td>
</tr><tr>
<td>ShortGPT</td>
<td>6.98</td>
<td>33.21</td>
<td>268.11</td>
</tr><tr>
<td rowspan="4">LLaMA-2-13B (PPL=4.57)</td>
<td>PGSVD</td>
<td>5.36</td>
<td>7.09</td>
<td>18.04</td>
</tr><tr>
<td>LLM-Pruner</td>
<td>5.57</td>
<td>12.19</td>
<td>32.20</td>
</tr><tr>
<td>SliceGPT</td>
<td>5.88</td>
<td>9.97</td>
<td>10.77</td>
</tr><tr>
<td>ShortGPT</td>
<td>5.40</td>
<td>30.48</td>
<td>187.23</td>
</tr></table>

Therefore,different tolerances are assigned to dif-ferent models, but within each model all layers share a uniform error tolerance,leading to hetero-geneous compression ratios across layers. By ad-justing only two hyperparameters (one for each modality), PGSVD enables an effective allocation of compression ratios for all layers in the network.

## 5.3 Comparison with Other Baseline Methods

Finally, we compare PGSVD with other baseline methods including LLM-Pruner (Ma et al.,2023), ShortGPT (Men et al., 2024),and SliceGPT (Ashk-boos et al., 2024). ShortGPT is a zero-shot com-pression method, whereas LLM-Pruner leverages gradient information for compression.

**Language** **Modelingg.** As shown in Table 3, de-spite not using gradient information and relying only on a small sample of data for covariance ap-proximation, PGSVD consistently achieves lower perplexity than LLM-Pruner and ShortGPT. Com-pared to SliceGPT (Ashkboos et al., 2024),PGSVD performs better at lower compression ratios, but is outperformed at the highest compression level (50%).

**Reasoning** **Tasks.** We also evaluated the accu-racy of these methods on multiple reasoning bench-marks, as reported in Table 4. Despite being applied in a zero-shot manner without any task-

Table 4: Reasoning performance of PGSVD versus prun-ing methods for 20% compression of LLaMA-2-7B.

<table border="1" ><tr>
<td>Task</td>
<td>Base</td>
<td>LLM-Pruner</td>
<td>SliceGPT</td>
<td>PGSVD</td>
</tr><tr>
<td>PIQA</td>
<td>78.07</td>
<td>75.95</td>
<td>61.26</td>
<td>71.27</td>
</tr><tr>
<td>WinoGrande</td>
<td>69.06</td>
<td>63.38</td>
<td>59.83</td>
<td>64.56</td>
</tr><tr>
<td>HellaSwag</td>
<td>76.00</td>
<td>67.83</td>
<td>44.28</td>
<td>60.96</td>
</tr><tr>
<td>ARC-e</td>
<td>76.30</td>
<td>64.31</td>
<td>46.09</td>
<td>70.75</td>
</tr><tr>
<td>ARC-c</td>
<td>43.34</td>
<td>39.93</td>
<td>28.41</td>
<td>36.52</td>
</tr><tr>
<td>Average</td>
<td>68.55</td>
<td>62.28</td>
<td>47.97</td>
<td>60.81</td>
</tr></table>

specific fine-tuning, PGSVD maintains competi-tive or superior accuracy compared to LLM-Pruner on most tasks, particularly on ARC-e and Wino-Grande,highlighting its robustness for reasoning-oriented evaluations. Although SliceGPT achieves stronger results on perplexity at extreme pruning ratios, its reasoning performance drops sharply at 20% compression, whereas PGSVD retains a bal-anced trade-off between compression and reason-ing accuracy. Overall, these results confirm that PGSVD preserves both generalization and reason-ing capabilities.

# 6 Conclusion

We have presented a framework for the compres-sion of LLMs/VLMs that integrates theory and practice. We have established a loss bound show-ing how layer compression influences overall net-work loss. By formulating rank selection as a bi-objective optimization problem, we have theoreti-cally demonstrated that a uniform error assignment across layers yields surrogate Pareto-optimal het-erogeneous ranks, simplifying rank search to a sin-gle knob. We have further proposed PGSVD to improve activation-aware compression via Pareto-guided rank selection. Empirically, PGSVD has shown consistent performance improvement in var-ious models, increasing accuracy by over 30% on LLM reasoning tasks. PGSVD has also been ef-fectively generalized to vision-language modeling, achieving up to 40% compression in zero-shot set-tings while preserving accuracy.

<!-- 8 -->

# 7 Limitations

The PGSVD algorithm relies on uniform-tolerance policies derived from robust assumptions. Future work may investigate learning tolerance thresh-olds directly from data, while further extensions involve coupling the framework with quantization, lightweight fine-tuning, and systematic evaluations on larger VLMs.

# References

Mistral AI. 2023. Mistral 7b: A 7-billion parameter language model. arXiv preprint arXiv:2310.06825.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gen-nari do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. In Inter-national Conference on Learning Representations (ICLR).

Yonatan Bisk, Rowan Zellers, Jianfeng Gao,and1 Yejin Choi. 2020. Piqa: Reasoning about physical com-monsense in natural language. In Proceedings of AAAI.

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx,Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emmma Brunskill, and et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.

Lukas Bossard, Matthieu Guillaumin,and Luc Van Gool. 2014. Food-101-mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), pages 446-461.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. 2014. De-scribing textures in the wild. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedin1gs of NAACL-HLT.

Peter Clark, Isaac Cowhey, Oren Etzioni,Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question an-swering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 2023. Spqr: A sparse-quantized rep-resentation for efficient generative inference. arXiv preprint arXiv:2306.03078.

Li Fei-Fei, Rob Fergus, and Pietro Perona. 2004. Learn-ing generative visual models from few training exam-ples: An incremental bayesian approach tested on 101object categories. CVPR Workshop on Generative-Model Based Vision.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. Sparsegpt: Massive language models can be accurately pruned in one-shot. arXiv preprint arXiv:2301.00774.

Elias Frantar, Eldar Kurtic, and Dan Alistarh. 2022. Op-timal brain compression: A framework for practical pruning of pretrained transformers. arXiv preprint arXiv:2208.11580.

Shangqian Gao, Ting Hua,Yen-Chang Hsu,Yilin Shen, and Hongxia Jin. 2024. Adaptive rank selections for low-rank approximation of language models. Pro-ceedings of the 2024 Conference of the North Amer-ican Chapter of the Association for Computational Linguistics.

Patrick Helber, Benjamin Bischke, Andreas Dengel,and Damian Borth. 2019. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, volume 12, pages 2217-2226.

Dan Hendrycks, Collin Burns,Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language under-standing. In Proceedings of ICLR.

Yen-Chang Hsu, Ting Hua, Sungen Chang,Qian Lou, Yilin Shen, and Hongxia Jin. 2022a. Language model compression with weighted low-rank factorization. International Conference on Learning Representa-tions (ICLR).

Yen-Chang Hsu,Ting Hua, Sungen Chang, Qian Lou, Yilin Shen,and Hongxia Jin. 2022b. Language model compression with weighted low-rank factor-ization. In International Conference on Learning Representations (ICLR).

Jonathan Krause, Michael Stark, Jia Deng,and Li Fei-Fei. 2013. 3d object representations for fine-grained categorization. In IEEE International Conference on Computer Vision Workshops (ICCVW).

Guokun Lai, Qizhe XXie,Hanxiao Liu,Yiming Yang, and Eduard Hovy. 2017. RACE:Large-scale read-ing comprehension dataset from examinations. In Proceedings of EMNLP.

Zhenzhong Lan,Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. Albert: A lite bert for self-supervised learning of language representations. International Confer-ence on Learning Representations (ICLR).

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C.H. Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research.

<!-- 9 -->

Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. Llm-pruner: On the structural pruning of large lan-guage models. In Advances in Neural Information Processing Systems (NeurIPS).

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin,Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853.

Stephen Merity, Caiming Xiong, James Bradbury,and Richard Socher. 2016. Pointer sentinel mixture mod-els. arXiv preprint arXiv:1609.07843.

Denis Paperno, Germán Kruszewski, Angeliki Lazari-dou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of ACL.

Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. 2012. Cats and dogs. In IEEE Con-ference on Computer Vision and Pattern Recognition (CVPR), pages 3498-3505.

David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Luis Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. 2021. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-try, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.Learn-ing transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 8748-8763.PMLR.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhaga-vatula,and Yejin Choi. 2020. Winogrande: An ad-versarial winograd schema challenge at scale. In Proceedings of AAAI.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled ver-sion of bert: Smaller, faster, cheaper, and lighter. arXiv:1910.01108.

Victor Sanh, Thomas Wolf, and Alexander M. Rush. 2020. Movement pruning: Adaptive sparsity by fine-tuning. Advances in Neural Information Processing Systems (NeurIPS).

Sheng Shen,Zhen Dong, Jiayu Ye, Linjian Ma,Zhewei Yao,Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. 2020. Q-bert: Hessian based ultra low pre-cision quantization of bert. AAAI Conference on Artificial Intelligence.

Dachuan Shi,Chaofan Tao, Ying Jin, Zhendong Yang, Chun Yuan, and Jiaqi Wang. 2023. Upop: Uni-fied and progressive pruning for compressing vision-language transformers. In International Conference on Machine Learning (ICML), Proceedings of MIa-chine Learning Research.

Masao Someki, Yifan Peng, Siddhant Arora, Markus Müller, Athanasios Mouchtaris, Grant Strimel, Jing Liu, and Shinji Watanabe. 2025. Context-aware dy-namic pruning for speech foundation models. In The Thirteenth International Conference on Learn-ing Representations.

Emma Strubell, Ananya Ganesh, and Andrew McCal-lum. 2019. Energy and policy considerations for deep learning in nlp. In Proceedings of the 57th An-nual Meeting of the Association for Computational Linguistics (ACL).

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowl-edge. In Proceedings of NAACL-HLT.

Hugo Touvron, Matthieu Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal,Eric Hambro,Au-rélien Azhar, Justin Rodriguez, Armand Joulin, and Edouard Grave. 2023. Llama 2: Open founda-tion and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Sys-tems (NeurIPS).

Xin Wang,Samiul Alam,Zhongwei Wan, Hui Shen,and Mi Zhang. 2025. SVD-LLM V2: Optimizing Sin-gular Value Truncation for Large Language Model Compression. In Proceedings of the 2025 Confer-ence of the Nations of the Americas Chapter of the Association forComputational Linguistics: Human Language Technologies (Long Papers), pages 4287-4296,Albuquerque, New Mexico. Association for Computational Linguistics.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. 2024. Svd-llm: Truncation-aware singular value de-composition for large language model compression. arXiv preprint arXiv:2403.07378.

Shuohang Yang, Xuguang Meng, Haoran Li, Jiahui Wang,Hongxia Tang,Shuo Lin, Xinyang Chen, Zheng Zhang, Weijia Liu, and Jingdong Wang. 2024. Ecoflap: Efficient coarse-to-fine layer-wise pruning for vision-language models. In Proceedings of the 41st International Conference on Machine Learning (ICML).

Qinghao Ye,Haiyang Xu, Guohai Xu,Jiabo Ye,Ming Yan,Yiyang Zhou,Junyang Wang,Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuan-hong Xu,Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. 2023. mplug-owl: Modularization empowers large lan-guage models with multimodality. arXiv preprint arXiv:2304.14178.

<!-- 10 -->

Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, and Guangyu Sun. 2023. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi,and Yejin Choi.2019.Hellaswag:Can a machine really finish your sentence? In Proceedings of ACL.

Zhiyuan Zhang,XuefeiNingSun,SongHan, JieTang, and Bolin Ding. 2024. Compressing transformers: Features are low-rank, but weights are not! In Pro-ceedings of the AAAI Conference on Artificial Intelli-gence,volume 38,pages 13714-13722.

<!-- 11 -->

# A Proof of Theorem 1

Proof. Let\|·\|be the operator (spectral) norm and \|·\|Fthe Frobenius norm. For batch size B, let Xl=[xl(1)⋯xl(B)]and define:

$$\boldsymbol {z}_{l}^{(i)}=\mathbf {W}_{l}\boldsymbol {x}_{l}^{(i)}$$

$$\mathbf {D}_{l}^{(i)}:=\text {diag}\left(σ^{\prime }\left(\boldsymbol {z}_{l}^{(i)}\right)\right)$$

$$\mathbf {J}_{l}^{(i)}:=\mathbf {D}_{l}^{(i)}\mathbf {W}_{l}$$

AssumeΔX1=0. Using the first-order expan-sion (dropping second-order terms) for each sample i we have:

$$\Delta \boldsymbol {x}_{l+1}^{(i)}\approx \mathbf {D}_{l}^{(i)}\left(\mathbf {W}_{l}\Delta \boldsymbol {x}_{l}^{(i)}+\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(i)}\right)$$

Assume\|Dl(i)\|≤c,then we have:

$$\left\|\Delta \boldsymbol {x}_{l+1}^{(i)}\right\|_{2}\leq \left\|\mathbf {J}_{l}^{(i)}\right\|\left\|\Delta \boldsymbol {x}_{l}^{(i)}\right\|_{2}+c\left\|\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(i)}\right\|_{2}.$$

Stack columns and define:

$$\mathbf {G}_{l}:=\left[\mathbf {J}_{l}^{(1)}\Delta \boldsymbol {x}_{l}^{(1)}\cdots \mathbf {J}_{l}^{(B)}\Delta \boldsymbol {x}_{l}^{(B)}\right],$$

$$\mathbf {Q}_{l}:=\left[\mathbf {D}_{l}^{(1)}\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(1)}\cdots \mathbf {D}_{l}^{(B)}\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(B)}\right]$$

ThenΔXl+1≈Gl+Qland:

$$\left\|\mathbf {G}_{l}\right\|_{F}^{2}=\sum _{i=1}^{B}\left\|\mathbf {J}_{l}^{(i)}\Delta \boldsymbol {x}_{l}^{(i)}\right\|_{2}^{2}$$

$$\leq \left(\sup _{i}\left\|\mathbf {J}_{l}^{(i)}\right\|^{2}\right)\sum _{i=1}^{B}\left\|\Delta \boldsymbol {x}_{l}^{(i)}\right\|_{2}^{2}$$

$$=\left(\sup _{i}\left\|\mathbf {J}_{l}^{(i)}\right\|\right)^{2}\left\|\Delta \mathbf {X}_{l}\right\|_{F}^{2}$$

$$\left\|\mathbf {Q}_{l}\right\|_{F}^{2}=\sum _{i=1}^{B}\left\|\mathbf {D}_{l}^{(i)}\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(i)}\right\|_{2}^{2}$$

$$\leq c^{2}\sum _{i=1}^{B}\left\|\Delta \mathbf {W}_{l}\boldsymbol {x}_{l}^{(i)}\right\|_{2}^{2}$$

$$=c^{2}\left\|\Delta \mathbf {W}_{l}\mathbf {X}_{l}\right\|_{F}^{2}$$

By triangle inequality in Frobenius norm we have:

$$\left\|\Delta \mathbf {X}_{l+1}\right\|_{F}\leq \left\|\mathbf {G}_{l}\right\|_{F}+\left\|\mathbf {Q}_{l}\right\|_{F}$$

$$\leq \mathcal {K}_{l}\left\|\Delta \mathbf {X}_{l}\right\|_{F}+c\left\|\Delta \mathbf {W}_{l}\mathbf {X}_{l}\right\|_{F}$$

whereKl:=supi\|Jl(i)\|and unrolling forl= 1,...,Lwe have:

$$\left\|\Delta \mathbf {X}_{L+1}\right\|_{F}\leq \sum _{l=1}^{L}\left(\prod _{m=l+1}^{L}\mathcal {K}_{m}\right)c\left\|\Delta \mathbf {W}_{l}\mathbf {X}_{l}\right\|_{F}$$

For a scalar lossL(Y)withY=XL+1,first-order Taylor and Cauchy-Schwarz yield:

$$\Delta \mathcal {L}\approx \left\langle \nabla _{\mathbf {Y}}\mathcal {L},\Delta \mathbf {Y}\right\rangle _{F}\quad \Rightarrow |\Delta \mathcal {L}|\leq \left\|\nabla _{\mathbf {Y}}\mathcal {L}\right\|_{F}\|\Delta \mathbf {Y}\|_{F}$$

☐

# B Proof of Proposition 1

Proof.(P⇒E)Take any feasible ranks{rl}l=1L for(P).Setϵl:=el(rl).Sincehl(ϵl)is the min-imal parameter count achieving error ≤ϵl,we havehl(ϵl)≤Pl(rl)for each l. Thus ∑lhl(ϵl)≤∑lPl(rl)≤b,SO{ϵl}is feasible for (E),and the (E) objective equals\sum_{l}\alpha_{l}\varepsilon_{l}=\sum_{l}\alpha_{l}\left(r_{l}\right),the surrogate objective in (P). Taking the infimum over feasible{rl}givesOPT(E)≤OPT(P)

(E⇒P)Let{ϵl★}be optimal for (E). For each l,pick rl★∈arg min{P():()≤ϵ★},So thatPl(rl★)=hl(ϵl★) andel(rl★)≤ϵl★.Then ∑lPl(rl★)=∑lhl(ϵl★)≤b,SO{rl★}is feasible for (P) ,and

OPPT(P)=∑l=1Lαlel(rl★)≤∑l=1Lαlϵl★=OPT(E)

HenceOPT(P)≤OPT(E). Combining both direc-tions yields equality. ☐

# C Proof of Lemma 1

Proof. Write∑l(·)for∑l=1L(·). By assumnption, $\underline {h},\bar {h}:[0,1]\rightarrow$ R≥0 are nonincreasing,convex, and bound eachhlpointwise. The budget range $L\bar {h}(1)\leq b\leq L\bar {h}(0)$ ensures feasibility.

(i) Symmetric surrogate $\bar {h}$  Consider miinϵ∈[0,1]L∑lϵl s.t. $\sum _{l}\bar {h}\left(\varepsilon _{l}\right)\leq b.$  This program is convex; Slater holds for interior b,hence KKT are necessary and sufficient. Let μ≥0be the multiplier for the coupling constraint, ul,vl≥0for0≤ϵl≤1,and $g_{l}\in \partial \bar {h}\left(\varepsilon _{l}\right)$ .KKT: (i) stationarity1+μgl-ul+vl=0,(ii) comple-mentary slackness $μ\left(\sum _{l}\bar {h}\left(\varepsilon _{l}\right)-b\right)=0,u_{l}\varepsilon _{l}=0,$ vl(ϵl-1)=0,(iii)primal/dual feasibility. Since $\bar {h}$ is strictly decreasing and b is nondegenerate, the coupling constraint is active, soμ&gt;0.If 0&lt;ϵl&lt;1thenul=vl=0and1+μgl=0. Monotonicity of $\partial \bar {h}$  (convexity) forces all interior ϵl to be equal. If some coordinates were at the boundary while others interior, stationarity with μ&gt;0and the monotonicity of $\partial \bar {h}$  would conflict with the active budget unless allϵlcoincide (up

<!-- 12 -->

to kinks where subgradients are intervals). Hence there is a uniform optimizerϵl=ϵ★determined by $L\bar {h}\left(\varepsilon ^{\star }\right)=b$ (unique on the active segment by continuity and strict decrease).

(ii) Minimax optimality. Let H be the admissible families with $\underline {h}\leq h_{l}\leq \bar {h}$ pointwise. For any {hl}∈Hand any3 $\sum _{l}h_{l}\left(\varepsilon _{l}\right)\leq \sum _{l}\bar {h}\left(\varepsilon _{l}\right)$ ,hence $\left\{\varepsilon :\sum _{l}\bar {h}\left(\varepsilon _{l}\right)\leq b\right\}\subseteq \left\{\varepsilon :\sum _{l}h_{l}\left(\varepsilon _{l}\right)\leq b\right\}$ .Mini-mizing the same linear objective over a smaller set yields a larger (optimal) value, so the worst case is attained at $h_{l}=\bar {h}.$  By (i), its optimizer is uniform. Thus the uniformn solution is minimax.

(iii) Bracketing. Define ϵu,ϵ\ell∈[0,1] by L $\bar {h}\left(\varepsilon ^{\mathrm {u}}\right)=b$ and $L\underline {h}\left(\varepsilon ^{\ell }\right)=b$ (use generalized inverses at flats). The uniform vector(ϵu,⋯,ϵu) is feasible for the true problem since $h_{l}\leq \bar {h},$ hence p≤αLϵu. For the lower bound, replace eachhl by $\underline {h}$ , which enlarges the feasible set; the relaxed problem is convex and, by the same KKT argument, has a uniform optimum ≈ϵ\ell,yieldingαLϵ\ell≤Combining givesαLϵ\ell≤p≤αLϵu ☐

# D Homogeneous Assumption

**Remark** **1** (Homogeneous sensitivities as a mini-max-robust surrogate). Recall problem (E) defined as 1minΩ0≤ϵl≤1∑lαlϵls.t. ∑lhl(ϵl)≤b.When a1 are uncertain, consider the robust counterpart

minϵmaxα∈U∑lαlϵl $\mathcal {U}=\left\{α:\underline {α}_{l}\leq α_{l}\leq \bar {α}_{l}\right\}$ 

$\text {Foranyfeasible}\varepsilon ,\max _{α\in \mathcal {U}}\sum _{l}α_{l}\varepsilon _{l}=\sum _{l}\bar {α}_{l}\varepsilon _{l}$ sinceϵl≥0 If the bounds are common $\left(\bar {α}_{l}=\bar {α}\right),$ this reduces exactly to the homogeneous-α objec-tive $\bar {α}\sum _{l}\varepsilon _{l}$  If they differ,letting $\bar {α}=\max _{l}\bar {α}_{l}$ gives $\sum _{l}\bar {α}_{l}\varepsilon _{l}\leq \bar {α}\sum _{l}\varepsilon _{l}$ so minimizing $\bar {α}\sum _{l}\varepsilon _{l}i$ minimax-safe.Thus, $α_{l}=\bar {α}$ provides aprincipled robust surrogate for (E).

# E Proof of Theorem 2

Proof. Fix a uniform toleranceϵ∈[0,1]and let rl(ϵ)be the minimal SVD rank per layer achieving ((ϵ))≤ϵ(byEYM).Set

$$H(\varepsilon ):=\sum _{l=1}^{L}P_{l}\left(r_{l}(\varepsilon )\right)$$

$$A(\varepsilon ):=\sum _{l=1}^{L}α_{l}e_{l}\left(r_{l}(\varepsilon )\right)$$

By the Rank-ϵAllocation Equivalence (Prop0.1), the constrained surrogate problem “minimize ∑lαlel(rl)subject to∑lPl(rl)≤b"is equiva-lent to its ε-formulation. Under the homogeneous sensitivity and bounded-profile assumption (The-orem 1), the optimal solution of this surrogate for any budget b is attained by a uniform tolerance across layers. Choosingb:=H(ϵ),the uniform choice at level 3 is therefore optimal among all feasible allocations with total parametetr≤H(ϵ) i.e.,there is no other feasible allocation that si-multaneously achieves∑lPl(rl)≤H(ϵ) and ∑lαlel(rl)&lt;A(ϵ).Hence(H(ϵ),A(ϵ))is(sur-rogate) Pareto-optimal for (B).

Ifε,ϵ′induce the same rank vector(rl(ϵ))l= (rl(ϵ′))l,thenH(ϵ)=H(ϵ′)andA(ϵ)=A(ϵ′) by definition, so they correspond to the same fron-tier point. ☐

# F Derivation of ALS Updates

We minimize

$$\min _{\mathbf {A},\mathbf {B}}\|\mathbf {WX}-\mathbf {ABX}\|_{F}^{2}\tag{7}$$

LetM=XX\topand using the Frobenius identity \|Y\|F2=tr(YY\top),we write

$$\text {tr}\left((\mathbf {W}-\mathbf {AB})\mathbf {M}(\mathbf {W}-\mathbf {AB})^{\top }\right)\quad =\text {tr}\left(\mathbf {WMW}^{\top }\right)-2\text {tr}\left(\mathbf {ABMW}^{\top }\right)+\text {tr}\left(\mathbf {ABMB}^{\top }\mathbf {A}^{\top }\right)$$

# Updlate for A.

$$\frac {\partial }{\partial \mathbf {A}}=-2\mathbf {BMW}^{\top }+2\mathbf {ABMB}^{\top }$$

Setting to zero:

$$\mathbf {A}=\mathbf {WMB}^{\top }\left(\mathbf {BMB}^{\top }\right)^{\dagger }.$$

# Update for B.

$$\frac {\partial }{\partial \mathbf {B}}=-2\mathbf {A}^{\top }\mathbf {WM}+2\mathbf {A}^{\top }\mathbf {ABM}.$$

Setting to zero:

$$\mathbf {B}=\left(\mathbf {A}^{\top }\mathbf {A}\right)^{\dagger }\mathbf {A}^{\top }\mathbf {W}$$

**ALS Updates.**

<table border="1" ><tr>
<td>A=WMB\top(BMB\top)\dagger,B=(A\topA)\daggerA\topW</td>
</tr></table>

(8)

<!-- 13 -->

<!-- 1.0 98% upper band 0.8 Un001a1eueled aNlaele 0.6 0.4 0.2 0.0 0.0 0.2 0.4 0.6 0.8 1.0 Relative error -->
![](./images/ab16233d164d857d.jpg)

<!-- 1.0 98% upper band 0.8 Un001aeueed aNiaeleX 0.6 0.4 0.2 0.0 0.0 0.2 0.4 0.6 0.8 1.0 Relative error -->
![](./images/6e1c932b1ca480f6.jpg)

Figure 4: SVD profiles for LLaMA-2 7B (left) and 13B (right).

Table 5: Zero-shot accuracy across tasks for LLaMA-2-7B and Mistral-7B at 20% and 40% compression (bold indicates the best accuracy per task among compressed methods).

<table border="1" ><tr>
<td>Model</td>
<td>Compression</td>
<td> Method</td>
<td>ARC-C</td>
<td>BoolQ</td>
<td>HellaSwag</td>
<td> MathQA</td>
<td>MMLU</td>
<td>Race</td>
<td>Avg</td>
</tr><tr>
<td rowspan="7">LLaMA-2-7B</td>
<td></td>
<td>Base</td>
<td>43.34</td>
<td>77.71</td>
<td>76.00</td>
<td>28.07</td>
<td>41.82</td>
<td>39.52</td>
<td>51.08</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD-LLM</td>
<td>34.39</td>
<td>63.09</td>
<td>59.19</td>
<td>25.33</td>
<td>26.20</td>
<td>34.55</td>
<td>40.46</td>
</tr><tr>
<td>SVD-ALS</td>
<td>33.70</td>
<td>68.01</td>
<td>59.55</td>
<td>24.96</td>
<td>26.64</td>
<td>33.97</td>
<td>41.14</td>
</tr><tr>
<td>PGSVD</td>
<td>36.52</td>
<td>67.43</td>
<td>60.96</td>
<td>25.76</td>
<td>27.48</td>
<td>35.02</td>
<td>42.20</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD-LLM</td>
<td>22.95</td>
<td>43.21</td>
<td>41.73</td>
<td>22.24</td>
<td>23.34</td>
<td>28.80</td>
<td>30.83</td>
</tr><tr>
<td>SVD-ALS</td>
<td>22.87</td>
<td>42.57</td>
<td>41.74</td>
<td>22.38</td>
<td>23.54</td>
<td>27.85</td>
<td>30.16</td>
</tr><tr>
<td>PGSVD</td>
<td>23.63</td>
<td>56.12</td>
<td>43.42</td>
<td>22.65</td>
<td>23.29</td>
<td>28.61</td>
<td>32.62</td>
</tr><tr>
<td rowspan="7">Mistral-7B</td>
<td></td>
<td>Base</td>
<td>50.43</td>
<td>83.61</td>
<td>81.08</td>
<td>35.71</td>
<td>59.64</td>
<td>40.86</td>
<td>58.56</td>
</tr><tr>
<td rowspan="3">20%</td>
<td>SVD-LLM</td>
<td>35.15</td>
<td>67.74</td>
<td>59.60</td>
<td>28.54</td>
<td>28.53</td>
<td>35.50</td>
<td>42.51</td>
</tr><tr>
<td>SVD-ALS</td>
<td>36.35</td>
<td>65.81</td>
<td>59.74</td>
<td>28.38</td>
<td>27.99</td>
<td>35.50</td>
<td>42.30</td>
</tr><tr>
<td>PGSVD</td>
<td>37.71</td>
<td>66.21</td>
<td>62.26</td>
<td>29.25</td>
<td>28.50</td>
<td>36.94</td>
<td>43.48</td>
</tr><tr>
<td rowspan="3">40%</td>
<td>SVD-LLM</td>
<td>21.67</td>
<td>38.62</td>
<td>37.87</td>
<td>22.18</td>
<td>23.38</td>
<td>27.08</td>
<td>28.47</td>
</tr><tr>
<td>SVD-ALS</td>
<td>20.82</td>
<td>38.41</td>
<td>37.87</td>
<td>22.75</td>
<td>23.28</td>
<td>27.18</td>
<td>28.39</td>
</tr><tr>
<td>PGSVD</td>
<td>21.42</td>
<td>39.02</td>
<td>39.12</td>
<td>22.35</td>
<td>23.44</td>
<td>27.46</td>
<td>28.80</td>
</tr></table>

# G Convex Envelope of SVD Profiles

The derivation of Pareto-guided ranks assumes that the SVD profiles (i.e., the rank-error relationships) are convexly bounded (Lemma 1). Figure 4 em-pirically supports this assumption. In Fig. 4, every single line corresponds to the SVD profile(h1)of a specific layer l of each network for all layers. It is observed that the relative SVD profiles naturally bounded by a convex upper bound. Although the uniform error-allocation strategy in PGSVD relies on the upper convex bound of the SVD profiles, the lower bound provides an indication of how close the achieved solution may be to the true optimum. As shown in Fig. 4, after excluding a small fraction of outlier layers, the lower bound of the remain-ing 98% of layers approach the upper bound, sug-gesting that the practical rank-error distributions closely follow the assumed convex envelope.

<!-- 14 -->

