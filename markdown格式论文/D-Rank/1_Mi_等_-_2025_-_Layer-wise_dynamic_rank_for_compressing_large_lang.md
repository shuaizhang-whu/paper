# LAYER-WISE DYNAMIC RANK FOR COMPRESSING

# LARGE LANGUAGE MODELS

**Zhendong Mi**

Stevens Institute of Technology

zmi2@stevens.edu

**Grace Li Zhang**

Technical University of Darmstadt

grace.zhang@tu-darmstadt.de **Bian Sun**

Carnegie Mellon University

bians@alumni.cmu.edu

**Shaoyi Huang***

Stevens Institute of Technology

shuang59@stevens.edu

## ABSTRACT

Large language models (LLMs) have rapidly scaled in size, bringing severe memory and computational challenges that hinder their deployment. Singular Value De-composition (SVD)-based compression has emerged as an appealing post-training compression technique for LLMs, yet most existing methods apply a uniform compression ratio across all layers, implicitly assuming homogeneous information included in various layers. This overlooks the substantial intra-layer heterogeneity observed in LLMs, where middle layers tend to encode richer information while early and late layers are more redundant. In this work, we revisit the existing SVD-based compression method and propose D-Rank, a framework with layer-wise balanced Dynamic **Rank** allocation for LLMs compression. We first introduce effective rank as a principled metric to measure the information density of weight matrices, and then allocate ranks via a Lagrange multiplier-based optimization scheme to adaptively assign more capacity to groups with higher information density under a fixed compression ratio. Moreover, we rebalance the allocated ranks across attention layers to account for their varying importance and extend D-Rank to latest LLMs with grouped-query attention. Extensive experiments on various LLMs with different scales and compression ratios demonstrate that D-Rank consistently outperforms baselines, achieving more than 15 lower perplexity on the C4 dataset withLLaMA-3-8B at 20% compression ratio and up to 5% higher zero-shot reasoning accuracy with LLaMA-7B at 40% compression ratio, while also delivering higher throughput.

## 1 INTRODUCTION

As large language models (LLMs) expand in both scale and deployment, their associated com-putational and environmental costs continue to escalate (Fernandez et al., 2025). For example, a 30B-parameter model (e.g., LLaMA-30B) reqjuires about 66GB for FP16 weights, which exceeds the capacity of a single GPU and forces the adoption of model parallelism across multiple GPUs (Touvron et al., 2023a). And a 176 billion parameter model BLOOM running on Google Cloud received 230,768 queries over 18 days, using an average of 40.32 kWh per day (roughly equivalent to 1,110 smartphone charges), demonstrating the substantial energy requirements for model inference at scale (Luccioni et al., 2024; 2023). To mitigate these costs, model compression techniques (e.g., pruning (Sun et al., 2024; Zhang et al., 2024; Ling et al., 2024; Frantar & Alistarh, 2023; Petri et al., 2023), quantization (Sun et al., 2023; Zhao et al., 2024; Xiao et al., 2023; Lin et al., 2024; Ashkboos et al., 2024b), and knowledge distillation (Gu et al., 2024; Magister et al., 2023; Jiang et al., 2023b; Qiu et al., 2024)) have been extensively employed to reduce computational and storage demands while preserving model accuracy, therefore facilitating more efficient LLM deployment. Although effective,these approaches typically require time-consuming retraining process and specialized hardware configurations (e.g., 2:4 semi-structured deployment for GPU-based pruning),creating practical deployment bottlenecks (Li et al., 2023; Ma et al., 2023).

*Corresponding author.

<!-- 1 -->

<!-- S70710t[0T.so] ZAzc9s7.60sZ.:AXTe -->

As an effective solution to these limmitations, compression techniques such as low-rank adaptation with Singular Value Decomposition (SVD) (Meng et al., 2024) have been extensively employed in LLM deployment (Balazy et al., 2025). In SVD-based low-rank adaptation, each weight matrix is approximated by decomposing it into three matrices of much smaller dimensions (Yuan et al., 2025; Wang et al., 2025b). After decomposition, matrix multiplications are performed on the lower-dimensional factors rather than the original full matrix, resulting in substantial paramneter reduction and improved storage and computational efficiency while preserving model performance comparable to the original full-rank model. Typically, a weight matrixW∈Rm×ncan be approximated as W≈UΣV\top,where&lt;min(mn)denotes the retained rank. A larger compression ratio will lead to a smaller k.

Despite the benefits and popularity of SVD-based low-rank adaptation in practical LLM deployment scenarios, there has been limited research on how to define an effective metric to quantify the information content in weight matrices, and subsequently attain dynamic ranks by leveraging the differences in information content between intra-layer attention matrices and cross-layer matrices, which therefore can maximize information preservation under a given overall compression ratio. In this work, we observe and identify several bottlenecks that hinder the efficiency of current SVD-based low-rank adaptation approaches: 1) limited effort has been devoted to designing effective metrics for measuring weight information content to determine optimal retained ranks k for each weight, leading to suboptimal compression performance; 2) existing approaches maintain uniform compression ratios across weight matrix types, failing to account for the substantial differences in their information density and inherent complexity; 3) in the latest LLMs with grouped-query attention, compression techniques such as grouping weight matrices across layers for joint compression may become ineffective due to substantial reduction in the column dimension of theWKandWVweight matrices compared with Multi-Head-Attention-based architectures (MHA), yet there is a lack of explanation for the underlying reasons as well as corresponding optimization strategies.

To address the bottleneck, we develop the layer-wise dynamic rank for SVD-based LLMs compres-sion. Specifically, we propose a metric, effective rank, for measuring information density of weight matrices. Subsequently, the effective rank will be employed to guide us in dynamically adjusting the retained rank for different types of weight matrices across different layers. To further improve the compression performance, we reallocate the preserved ranks across matrix types for attention layers by transferring part of the budget from matrices with lower information density to ones with higher information density while the same the overall target compression ratio. The main contributions of this work can be summarized as follows:

·We propose **D-Rank,** a layer-wise Dynamic Rank allocation approach for compressing LLMs. This approach enables us to preserve more information in large language models under the same compression budget, thereby achieving superior compression performance.

·We introduce a novel metric, effective rank, to quantify the information density of each grouped layer in LLMs. Moreover, we develop a Lagrangian multiplier-based framework that dynamically allocates ranks across grouped layers according to their effective rank, aiming to improve the information preserved in the models.

·Through effective rank analysis, we discover that the effective rank distribution among the attentions matrices is highly unbalanced:WQWKhave lower effective rank (less information) thanWV matrix. To address this issue, we propose a reallocation strategy that transfers part of the preserved rank budget fromWQWKWV

·Moreover, we analyze the reason why the performance of latest models (e.g., LLaMA-3) with grouped-query attention degrades in the state-of-the-art works, and we further demonstrate the effectiveness of D-Rank on the models.

Extensive experiments on the LLaMA, LLaMA-2,LLaMA-3,and Mistral families show that D-Rank consistently outperforms baselines, achieving more than 15 lower perplexity on LLaMA-3-8B model with 20% compression ratio on C4 datasets, and up to 5% higher accuracy on zero-shot reasoning tasks with LLaMA-7B model at 40% compression ratio, while it has even higher token throughput compared to baselines.

<!-- 2 -->

<!-- 2 MOTIVATION AND RESEARCH QUESTIONS -->

Recent research (Hu et al., 2025; Gao et al., 2024; Razzhigaev et al., 2023; Wei et al., 2024) show that the **information** **content** of weight matrices varies significantly across layers. For example, studies show that with respect to the input activations X, early and late layers of LLMs exhibit lower information density, while middle layers contain substantially richer information, formming a characteristic U-shaped distribution across depth (Razzhigaev et al., 2023; Hu et al., 2025).Although layer-wise information differences in LLMs have been discussed in other applications, in model compression, few works have investigated how to design metrics which can effectively quantify such differences across different weight matrices, and how to leverage these metrics to develop effective allocation strategies for layer-wise rank allocation. Then we naturally raise the following question:

## Question 1

How does **theinformation** **content** in weights vary across layers? **What** **metric** should we use to quantify it, and how can it guide adaptive rank allocation for model compression?

Prior work demonstrates that attention layers in Transformer-based models exhibit substantial re-dundancy and notable inter-layer heterogeneity (Voita et al., 2019). Additionally, different attention matrices in Transformer-based models show extremely unbalanced importance during fine-tuning, indicating that the effective parameter space varies significantly across different matrices in attention layers (Yao et al., 2024). Recently, several parameter efficient fine-tuing (PEFT) works (Zhang et al., 2023; Liu et al., 2024b) tend to allocate ranks or parameter budgets adaptively across layers and individual attention matrices across attention layers, consistently outperforming uniform rank allocation, and empirically demonstrating that different matrices possess varying levels of importance. However, most existing SVD-based model compression works apply identical compression ratios (or ranks) to all attention layer weight matrices, with limited exploration of inter-layer heterogeneity for adaptive compression ratio distribution. This motivates the following question:

## Question 2

Do different weight matrices in attention layers,especiallyWQ,WK,,andWV,contain different levels of information, and should non-uniform ranks be allocated for attention layer compression?

## 3 METHODOLOGY

### 3.1 NOTATION AND PRELIMINARY

Assume an LLM model with N layers and G groups, and for each group there are n layers,the weight ofi-thlayer inside of a group is denoted asW(i)∈R1x2.We can first concatenate matrices within the same group horizontally:W=[\begin{array}lllW(1)&W(2)&⋯&W()\end{array}]∈R1x(2).We then perform SVD:W=UΣV\top. After truncating to the top ksingular values, we obtain:W≈Wk=UkΣkVk\top, whereU∈Rd1xΣ∈RxV\top∈Rxnd2.We defineB=UΣ∈Rd1×as the shared basis matrix and splitVk\top into blocksC(i)∈Rkxd2as the layer-specific coefficient matrices: W(i)≈BC(i). That is, each column ofW(i) is expressed as a linear combination ofk shared basis vectors:W:,j(i)≈∑m=1kB:,mCm,j(i)

However, directly applying SVD on the weight matrix without considering the effects of the calibra-tion data on activation X is impractical since this might lead to significant compression loss and poten-tially affect the performance of the LLM after compression (Wang et al., 2025a). Therefore, several works (Yuan et al., 2025; Wang et al., 2025b) propose that we can incorporate the input activation sta-tistical information S for SVD calculation, which can be expressed asSS\top=coky(X\topX)and W=S-1(SW).Following these works (Wang et al.,2025a;b), we apply SVD to the scaled matrix SWinafW:SW≈Uk′Σk′Vk′\top, and we can reconstruct aW≈S-1Uk′Σk′Vk′\top=B′′C′whereB′′=S-1Uk′Σk′is thesharedbasis matrix andC′are the coefficient matrices.Notably,when n=1only, the procedure is the standard SVD-LLM approach.

<!-- 3 -->

<!-- nd2 nd2 d1 group 1 S1W1 W1 Effective Rank group 2 $\left\{\begin{matrix}\mathcal {R}_{\epsilon ff}(1)&grouI\\ \frac {LagrangeMuhtiliers}{k_{0}\times \sqrt {\mathcal {R}_{\epsilon ff}(k)}}{k_{0}\times \sqrt {\mathcal {R}_{\epsilon ff}(k)}}\\ \cdots &\frac {k_{0}grangeMuhtiliers}{k_{0}\times \sqrt {\mathcal {R}_{\epsilon f}(g)}}\\ \cdots &\frac {k_{0}grank_{1},k_{2},\ldots ,k_{g}rank_{k}}{forcorresondinggrou}&\frac {k_{1}}{S_{k}W_{\epsilon }\cup Σ_{k}V_{k}}\\ G\end{matrix}\right.\begin{pmatrix}\frac {Recoww.W$ {\begin{array}lRecoverW2s⋯sbbbbbbbbbbbb lef S2W2 Calculation -... W2 groupG SGWG WG -->
![](./images/0da1117d5cd33ac1.jpg)

Figure 1: The overall pipeline of our proposed D-Rank

3.2 LAYER-WISE DYNAMIC RANK SELECTION VIA EFFECTIVE RANK-BASED INFORMATION DENSITY CALCULATION

#### 3.2.1 EFFECTIVE RANK FORMULATION

Consider theg-th group of matrices denoted asWg∈Rd1×nd2.The effective rank ofWgis calculated based on the spectral entropy of the scaled matrixSgWg. We first calculate the i-th squared singular value ofSgWgasλgi=(σgi)2(0≤i&lt;1),which represents the energy along the i-thprincipal component. These squaredsingular values are then normalized to form a probability distribution P,and the i-thelement of the distribution is defined as:

$$p_{g}^{i}=\frac {λ_{g}^{i}}{\sum _{j}λ_{g}^{i}}\tag{1}$$

We further define the effective rankReffto evaluate the sensitivity of group gusing the exponential of the Shannon entropy of the distribution, which measures the number of significant singular values of the scaled matrixSgWg.We formulate the effective rank as follows:

$$\mathcal {R}_{\mathrm {eff}}(g)=\exp \left(-\sum _{i}p_{g}^{i}\log p_{g}^{i}\right)\tag{2}$$

The formulation considers the overall singular value distribution of each scaled matrixSgWg,which can be regarded as the information density of it. We use the effective rankR(g)to represent the minimum number of singular values required to effectively represent the uncompressed scaled matrix SgWg. A lower effective rank indicates higher redundancy, while a higher effective rank suggests higher information density of the group.

#### 3.2.2 RANK ALLOCATION VIA LAGRANGE MULTIPLIERS

**Motivation.** Table 1 shows that the effective rank varies sub- Table 1: Effective rank of grouped stantially across different layer groups, indicating the non- matrices for V,K,Q in LLaMA-uniform information density over depth. In particular, the mid- 7B on Wikitext-2 (two layers as a dle layers generally have higher effective ranks than the earlier group) and the later layers, which is consistent with existing studies showing the U-shaped information distribution across depth in Transformer-based models(Razzhigaev et al., 2023; Hu et al., 2025). Such variability implies that applying a single, uniform compression ratio to all groups may be suboptimal, as it ig-nores the depth-wise information density difference. For better performance with a fixed overall compression ratio,guided by effective rank, we allocate each group's retained rankkgbased on our proposed **rank allocation via Lagrangian multipliers.**

<table border="1" ><tr>
<td>Group Index</td>
<td>V</td>
<td>K</td>
<td>Q</td>
</tr><tr>
<td>1</td>
<td>118</td>
<td>6</td>
<td>7</td>
</tr><tr>
<td>3</td>
<td>592</td>
<td>8</td>
<td>12</td>
</tr><tr>
<td>7</td>
<td>778</td>
<td>12</td>
<td>33</td>
</tr><tr>
<td>10</td>
<td>1026</td>
<td>15</td>
<td>24</td>
</tr><tr>
<td>12</td>
<td>973</td>
<td>12</td>
<td>25</td>
</tr><tr>
<td>14</td>
<td>1148</td>
<td>11</td>
<td>29</td>
</tr><tr>
<td>16</td>
<td>846</td>
<td>10</td>
<td>23</td>
</tr></table>

This reallocation maximizes the information preserved in the model after compression under a fixed target compression ratio. And the group with a higher effective rankReff(g) will be assigned a larger budget proportionally.

Suppose a LLM model with N layers and G groups, and for each group there are n layers.For the i-thgroup ,we have Reff(g) as the effective rank to quantify the information density of the group. We

<!-- 4 -->

denote was the parameter cost per rank to represent the number of parameters required to increase the rank of the group by one. For a shared basis, this is calculated asω=1+n2,where n is the number of layers in the group. We usegto denote the number of ranks to be allocated to group g. We further define a total reallocation error as\elltt, which penalizes distribution inconsistency between the allocated rank and the effective rank accumulated across all groups, under the assumption that the error is inversely proportional to the allocated rank and proportional to the effective rank. Suppose that the total number of parameters of all groups is T and the target compression ratio is θ,we denoteTbug=T(1-θ)=∑g=1kgωas the total number of parameters in the compressed module. We then formulate the optimization problem as follows:

subject to ∑g=1kgω=Tbudget

$$\underset {k_{1},k_{2},\cdots }{\text {minimize}}\quad \ell _{\text {total}}=\sum _{g=1}\frac {\mathcal {R}_{\mathrm {eff}}(g)}{k_{g}}\tag{3}$$

Using Lagrange multipliers, we can solve the constrained optimization problem with the following Lagrange function:

$$\mathcal {F}\left(\left\{k_{g}\right\},λ\right)=\sum _{g=1}\frac {\mathcal {R}_{\text {eff}}(g)}{k_{g}}+λ\left(\sum _{g=1}k_{g}ω-\mathcal {T}_{\text {budget}}\right)\tag{4}$$

λ is the Lagrangian multiplier. Taking the derivative of F with respect to eachkgand setting it to zero:

$$\frac {\partial \mathcal {F}}{\partial k_{g}}=-\frac {\mathcal {R}_{\mathrm {eff}}(g)}{k_{g}^{2}}+λω=0\tag{5}$$

The solution reveals that the optimal rankkgfor each group should be determined according to the following proportionality:

$$k_{g}\propto \sqrt {\frac {\mathcal {R}_{\mathrm {eff}}(g)}{ω}}\tag{6}$$

We can see that the optimal rank is proportional to the square root of the group'sReff(g)and inversely proportional to the square root of its parameter cost (more expensive groups get fewer ranks). Applying the budget constraint,we have

$$k_{g}=\frac {\mathcal {T}_{\text {budget}}}{\sum _{j=1}\sqrt {\mathcal {R}_{\text {eff}}(j)ω}}·\frac {\sqrt {\mathcal {R}_{\text {eff}}(g)}}{\sqrt {ω}},\tag{7}$$

Afterwards, we obtain a listL that records the retained rank required for each group of such weight matrices[k_{1},k_{2},\ldots,right]. Detailed allocation strategy is shown in Appendix A.3. The layer-wise dynamic rank selection pipeline is illustrated in Figure 1. First, for each group, weight matrices across n layers are concatenated horizontally and multiplied by S to form scaled matrixSgWg(Si calculated bySS\top=chlk(X\topX)from activations X and g is the index of the group), then we calculate the effective rank of each group of scaled matrix. After we get the rank{k1k2⋯kG}for each grouped matrix with Lagrange Multiplier, we will use them as the singular values to perform the SVD compression for every scaled weight matrix.

### 3.3 BALANCING DYNAMIC RANK ACROSS ATTENTION LAYERS

**Motivation.**We group every two layers of LLaMA-7B and estimate the effective ranks of WQ,WK,WV,as shown in Figure 2. We ob-serve thatWVconsistently exhibits much largerRff (which is ofter&gt;1000)thanWQ,WKindicating that the information density is unevenly distributed across the attention layers. This observation moti-vates us to consider two key questions: do different weight matrices show substantial disparities in infor-mation density, and can such disparities inform how we adjust compression ratios across matrices?

<!-- WQ O 1000 WK YuEJ aAoalo PalguUS WV 750 500 250 0 、 2 34 5 6 7 8 9 10 11 12 13 14 15 16 Layer group (every 2 layers as a group) -->
![](./images/cb00ddfa6a9bfae5.jpg)

Figure 2: Effective ranks of grouped WQ,WK,WVmatrices for LLaMA-7B model on Wikitext-2 (two layers as a group)

<!-- 5 -->

As discussed in the previous section, the value ofRffrepresents the minimum number of top-k singular values to effectively represent the matrix. Therefore, assigning the number of retained singular values k solely based on the effective ranks of each type of matrix according to the Lagrangian method would be unfair to theWVmatrices.

To address this, after computing the number of retained singular values k for each group of the WQWK,andWVmatrices using the Lagrangian allocation method, we reallocate part of the k originally assigned to theWQandWKmatrices to theWVmatrices. Suppose for a LLaMA-7B models (32 layers in total) has 4 groups, then each group has 8 layers. Based on our proposed rank allocation via Lagrange multipliers, we can obtain the listC of reallocated rank Wkfor each group ofWQWKWV as follows:

LQ=[k1Q,k2Q,k3Q,k4Q] LK=[k1K,k2K,k3K,k4K] LV=[k1V,k2V,k3V,k4V] (8)

We then define **an** **adjustment ratio** β and extract a portionof the rank proportional to β from the LQandLK,respectively. Then we sum up the extracted rank, and redistribute the accumulated rank evenly across the elements ofLV:

$$\begin{array}{l}\mathcal {L}_{\text {final-}k}^{Q}=(1-\beta )\left[k_{1}^{Q},k_{2}^{Q},k_{3}^{Q},k_{4}^{Q}\right],\\ \mathcal {L}_{\text {final-}k}^{K}=(1-\beta )\left[k_{1}^{K},k_{2}^{K},k_{3}^{K},k_{4}^{K}\right],\end{array}\tag{9}\tag{10}$$

$$t=\frac {\beta }{4}\left(\sum _{i=1}k_{i}^{Q}+\sum _{i=1}k_{i}^{K}\right)\tag{11}$$

$$\mathcal {L}_{\text {final-}k}^{V}=\left[k_{1}^{V}+t,k_{2}^{V}+t,k_{3}^{V}+t,k_{4}^{V}+t\right]\tag{12}$$

Then,we obtain the finaladjusted numbers of retained singular values, for theWQ,WK,and WV matrices in the attention layers. Overall, since theWVmatrices generally exhibit higher effective ranks than theWQandWKmatrices, this adjustment allows theWVmatrices,which require more information capacity, to retain higher singular values. The parameterβ serves as a tunable hyperparameter, and we will provide a detailed analysis of its impact in the experimental section.

### 3.4 DYNAMIC RANK ALLOCATION FOR MODELS WITH GROUPED-QUERY ATTENTION

We observe that when the number of layers in each group increases, there is a trend that the Table 2: Evaluation of PPL(↓) of LLaMA-3-8B on Wikitext-2 under 20% and 30% compression ratio performance will decrease (i.e.,ppl increases) on LLaMA-3, as shown in Table 2. We analyze the reasons as follows: 1) On LLaMA-3-8B, theWK,WVprojection matrices have dimen-sions of4096×1024.. When such matrices are horizontally concatenated within a group, the dimension expands severely and the matrix rank could be even larger than the rank of any indi-

<table border="1" ><tr>
<td>Method</td>
<td>Grouped layers</td>
<td>20%</td>
<td>30%</td>
</tr><tr>
<td>SVD-LLM</td>
<td>1</td>
<td>15.45</td>
<td>30.59</td>
</tr><tr>
<td rowspan="4">Basis Sharing</td>
<td>2</td>
<td>14.70</td>
<td>31.87</td>
</tr><tr>
<td>3</td>
<td>20.28</td>
<td>55.29</td>
</tr><tr>
<td>4</td>
<td>22.57</td>
<td>66.94</td>
</tr><tr>
<td>5</td>
<td>17.09</td>
<td>44.09</td>
</tr></table>

vidual matrix. Under a fixed compression ratio, the resulting SVD truncation will lead to a larger reconstruction error for the concatenated matrix than for compressing the original per-layer matrices separately.2) Since theWK,WVprojections in LLaMA-3 are architecturally slimmed to reduce KV-cache memnory compared to LLaMA and LLaMA-2 (Touvron et al., 2023a),grouping&gt;1layers for joint SVD results in fewer retained ranks per matrix under a fixed global compression ratio, leading to more aggressive compression of individual matrices. For example, at a 20% compression ratio,n=1retainsk=655ranks per group,while=yieldsk=109group ranks (around 546per matrix) (Wang et al., 2025a), demonstrating the more aggressive per-matrix compression.

To address the issue, for models with grouped-query attentions, such as LLaMA-3, we set the group size asn=1, and we use our proposed compression scheme that (i) dynamically adjusts the retained rank k of each layer according to its effective rank; (ii) reallocates a portion of the k budget from the WQandWKmatrices to theWVmatrices. Our experimental results demonstrate that the proposed method remains effective on LLaMA-3 architecture models.

<!-- 6 -->

Table 3: Comparison of PPL(↓) and Zero-shot(↑) performance of LLaMA-7B with baselines. The S of all tasks is obtained with the dataset **WikiText-2** **and**n=2

<table border="1" ><tr>
<td>RATIO</td>
<td>Method</td>
<td>WikiText-2↓</td>
<td>PTB↓</td>
<td>C4↓</td>
<td>Openb.↑</td>
<td>ARCe↑</td>
<td>WinoG.↑</td>
<td>HellaS.↑</td>
<td>ARC↑</td>
<td>PIQA↑</td>
<td>MathQA↑</td>
<td>Average*↑</td>
</tr><tr>
<td>0%</td>
<td>Original</td>
<td>5.68</td>
<td>8.35</td>
<td>7.34</td>
<td>0.28</td>
<td>0.67</td>
<td>0.67</td>
<td>0.56</td>
<td>0.38</td>
<td>0.78</td>
<td>0.27</td>
<td>0.47</td>
</tr><tr>
<td rowspan="6">20%</td>
<td>SVD</td>
<td>20061</td>
<td>20306</td>
<td>18800</td>
<td>0.14</td>
<td>0.27</td>
<td>0.51</td>
<td>0.26</td>
<td>0.21</td>
<td>0.53</td>
<td>0.21</td>
<td>0.31</td>
</tr><tr>
<td>FWSVD</td>
<td>1727</td>
<td>2152</td>
<td>1511</td>
<td>0.15</td>
<td>0.31</td>
<td>0.50</td>
<td>0.26</td>
<td>0.23</td>
<td>0.56</td>
<td>0.21</td>
<td>0.32</td>
</tr><tr>
<td>ASVD</td>
<td>11.14</td>
<td>16.55</td>
<td>15.93</td>
<td>0.25</td>
<td>0.53</td>
<td>0.64</td>
<td>0.41</td>
<td>0.27</td>
<td>0.68</td>
<td>0.24</td>
<td>0.43</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.94</td>
<td>18.05</td>
<td>15.93</td>
<td>0.22</td>
<td>0.58</td>
<td>0.63</td>
<td>0.43</td>
<td>0.29</td>
<td>0.69</td>
<td>0.24</td>
<td>0.44</td>
</tr><tr>
<td>Basis Sharing</td>
<td>7.74</td>
<td>17.35</td>
<td>15.03</td>
<td>0.28</td>
<td>0.66</td>
<td>0.66</td>
<td>0.46</td>
<td>0.36</td>
<td>0.71</td>
<td>0.25</td>
<td>0.48</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>7.45</td>
<td>15.99</td>
<td>13.73</td>
<td>0.29</td>
<td>0.69</td>
<td>0.66</td>
<td>0.47</td>
<td>0.36</td>
<td>0.72</td>
<td>0.25</td>
<td>0.49</td>
</tr><tr>
<td rowspan="6">30%</td>
<td>SVD</td>
<td>13103</td>
<td>17210</td>
<td>20871</td>
<td>0.13</td>
<td>0.26</td>
<td>0.51</td>
<td>0.26</td>
<td>0.21</td>
<td>0.54</td>
<td>0.22</td>
<td>0.30</td>
</tr><tr>
<td>FWSVD</td>
<td>20127</td>
<td>11058</td>
<td>7240</td>
<td>0.17</td>
<td>0.26</td>
<td>0.49</td>
<td>0.22</td>
<td>0.22</td>
<td>0.51</td>
<td>0.19</td>
<td>0.30</td>
</tr><tr>
<td>ASVD</td>
<td>51</td>
<td>70</td>
<td>41</td>
<td>0.18</td>
<td>0.43</td>
<td>0.53</td>
<td>0.37</td>
<td>0.25</td>
<td>0.65</td>
<td>0.21</td>
<td>0.38</td>
</tr><tr>
<td>SVD-LLM</td>
<td>9.56</td>
<td>29.44</td>
<td>25.11</td>
<td>0.20</td>
<td>0.48</td>
<td>0.59</td>
<td>0.40</td>
<td>0.26</td>
<td>0.65</td>
<td>0.22</td>
<td>0.40</td>
</tr><tr>
<td>Basis Sharing</td>
<td>9.25</td>
<td>29.12</td>
<td>22.46</td>
<td>0.27</td>
<td>0.63</td>
<td>0.63</td>
<td>0.40</td>
<td>0.30</td>
<td>0.68</td>
<td>0.24</td>
<td>0.45</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>8.97</td>
<td>26.40</td>
<td>20.44</td>
<td>0.28</td>
<td>0.65</td>
<td>0.64</td>
<td>0.42</td>
<td>0.32</td>
<td>0.69</td>
<td>0.25</td>
<td>0.46</td>
</tr><tr>
<td rowspan="6">40%</td>
<td>SVD</td>
<td>52489</td>
<td>59977</td>
<td>47774</td>
<td>0.15</td>
<td>0.26</td>
<td>0.52</td>
<td>0.26</td>
<td>0.22</td>
<td>0.53</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>FWSVD</td>
<td>18156</td>
<td>20990</td>
<td>12847</td>
<td>0.16</td>
<td>0.26</td>
<td>0.51</td>
<td>0.26</td>
<td>0.22</td>
<td>0.53</td>
<td>0.21</td>
<td>0.30</td>
</tr><tr>
<td>ASVD</td>
<td>1407</td>
<td>3292</td>
<td>1109</td>
<td>0.13</td>
<td>0.28</td>
<td>0.48</td>
<td>0.26</td>
<td>0.22</td>
<td>0.55</td>
<td>0.19</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM</td>
<td>13.11</td>
<td>63.75</td>
<td>49.83</td>
<td>0.19</td>
<td>0.42</td>
<td>0.58</td>
<td>0.33</td>
<td>0.25</td>
<td>0.60</td>
<td>0.21</td>
<td>0.37</td>
</tr><tr>
<td>Basis Sharing</td>
<td>12.39</td>
<td>55.78</td>
<td>41.28</td>
<td>0.22</td>
<td>0.52</td>
<td>0.61</td>
<td>0.35</td>
<td>0.27</td>
<td>0.62</td>
<td>0.23</td>
<td>0.40</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>11.99</td>
<td>56.04</td>
<td>37.22</td>
<td>0.23</td>
<td>0.57</td>
<td>0.61</td>
<td>0.36</td>
<td>0.27</td>
<td>0.64</td>
<td>0.23</td>
<td>0.42</td>
</tr><tr>
<td rowspan="6">50%</td>
<td>SVD</td>
<td>131715</td>
<td>87227</td>
<td>79815</td>
<td>0.16</td>
<td>0.26</td>
<td>0.50</td>
<td>0.26</td>
<td>0.23</td>
<td>0.52</td>
<td>0.19</td>
<td>0.30</td>
</tr><tr>
<td>FWSVD</td>
<td>24391</td>
<td>28321</td>
<td>23104</td>
<td>0.12</td>
<td>0.26</td>
<td>0.50</td>
<td>0.26</td>
<td>0.23</td>
<td>0.53</td>
<td>0.20</td>
<td>0.30</td>
</tr><tr>
<td>ASVD</td>
<td>15358</td>
<td>47690</td>
<td>27925</td>
<td>0.12</td>
<td>0.26</td>
<td>0.51</td>
<td>0.26</td>
<td>0.22</td>
<td>0.52</td>
<td>0.19</td>
<td>0.30</td>
</tr><tr>
<td>SVD-LLM</td>
<td>23.97</td>
<td>150.58</td>
<td>118.57</td>
<td>0.16</td>
<td>0.33</td>
<td>0.54</td>
<td>0.29</td>
<td>0.23</td>
<td>0.56</td>
<td>0.21</td>
<td>0.33</td>
</tr><tr>
<td>Basis Sharing</td>
<td>20.00</td>
<td>126.35</td>
<td>88.44</td>
<td>0.18</td>
<td>0.42</td>
<td>0.57</td>
<td>0.31</td>
<td>0.23</td>
<td>0.58</td>
<td>0.22</td>
<td>0.36</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>19.82</td>
<td>126.10</td>
<td>80.69</td>
<td>0.20</td>
<td>0.46</td>
<td>0.58</td>
<td>0.32</td>
<td>0.24</td>
<td>0.58</td>
<td>0.22</td>
<td>0.37</td>
</tr></table>

## 4 EXPERIMENTS

### 4.1 EXPERIMENTAL SETTING

**Datasets.** For language modeling, we use three datasets: PTB,WikiText2, and C4 ((Marcus et al., 1993);(Merity et al., 2017); (Raffel et al., 2020)). To evaluate the model's reasoning ability,we employ seven reasoning datasets: MathQA, PIQA, ARC-e,ARC-c,HellaSwag,WinoGrande,and OpenbookQA ((Amini et al., 2019); (Bisk et al., 2019); (Clark et al., 2018);(Zellers et al., 2019); (Sakaguchi et al., 2021); (Banerjee et al., 2019)). The LM-Evaluation-Harness framework has been applied to test every reasoning task through a zero-shot setting (Sutawika et al., 2024).

**Models.** We conduct comprehensive evaluations of D-Rank across multiple LLMs, including the LLaMA family (LLaMA-7B,LLaMA-13B, LLaMA-30B,LLaMA-2-7B,LLaMA-3-8B)((Touvron et al., 2023a); (Touvron et al., 2023b); (Dubey et al., 2024)) and Mistral-7B((Jiang et al., 2023a)).

**Baselines.** We contrast comparative evaluations with existing methods that utilize SVD-based weight approximation in individual layers without cross-layer parameter sharing. We specifically benchmark against FWSVD (Hsu et al., 2022), ASVD (Yuan et al., 2025), SVD-LLM (Wang et al., 2025b), and Basis Sharing (Wang et al.,2025a).

**Implementation Details and Hyperparameter Settings.** All experiments are conducted on two NVIDIA A100 80GB GPUs. The LLaMA-30B model is implemented in FP16 precision,while all other models utilize FP32 precision. We use FP64 to maintain the computational precision of matrix S. Matrix S is derived from 256 samples of WikiText-2 with a sequence length of 2048. Note that when the compression ratio is 40% or more, accumulated compression errors lead to significant inter-layer input deviation from original values. We adaptively update the downstream layer weights using the deviated inputs, similar to the method used in SVD-LLM.Following (Wang et al., 2025a), matrices like WQ,WKWVWup, and Wgate in MHA-based models are grouped and compressed in our experiments whenn&gt;1,whileWdownandWOare not grouped.

### 4.2 MAIN RESULTS

**Performance on generation and reasoning datasets.** On LLaMA-7B with S from Wikitext-2and group size=2,D-Rank consistently has a better performance under 20-50% compression compared with baselines as shown in Table 3. Compared with SVD-LLM, we reduce PPL on Wikitext-2, PTB and C4 by 6-32% across ratios. For instance, at 20% compression ratio D-Rank can achieve about 0.5 lower PPL than SVD-LLM and raise average zero-shot accuracy by about 0.11 at 30% compression ratio. Compared with Basis Sharing, our approach attains equal or higher average accuracy at all ratio and typically lower PPL on Wikitext-2 and C4, with a single notable

<!-- 7 -->

Table 4: PPL(↓) and1 Zero-shot(↑) performance on LLaMA-3-8B under 20% compression ratio. The S of all tasksis obtained with the dataset WikiText-2. For Basis sharing baseline,=5

<table border="1" ><tr>
<td>Method</td>
<td>WikiText-2↓</td>
<td>C4↓</td>
<td>Openb.↑</td>
<td>ARCe↑</td>
<td>WinoG.↑</td>
<td> HellaS.↑</td>
<td>ARCc↑</td>
<td>PIQA↑</td>
<td>MathQA↑</td>
<td>Average*↑</td>
</tr><tr>
<td>Original</td>
<td>6.14</td>
<td>9.47</td>
<td>0.34</td>
<td>0.75</td>
<td>0.70</td>
<td>0.57</td>
<td>0.40</td>
<td>0.79</td>
<td>0.27</td>
<td>0.55</td>
</tr><tr>
<td>FWSVD</td>
<td>4782</td>
<td>8195</td>
<td>0.01</td>
<td>0.04</td>
<td>0.01</td>
<td>0.02</td>
<td>0.01</td>
<td>0.02</td>
<td>0.01</td>
<td>0.02</td>
</tr><tr>
<td>ASVD</td>
<td>17.55</td>
<td>77.25</td>
<td>0.20</td>
<td>0.59</td>
<td>0.61</td>
<td>0.41</td>
<td>0.28</td>
<td>0.68</td>
<td>0.24</td>
<td>0.43</td>
</tr><tr>
<td>SVD-LLM</td>
<td>15.45</td>
<td>78.01</td>
<td>0.24</td>
<td>0.63</td>
<td>0.62</td>
<td>0.40</td>
<td>0.30</td>
<td>0.68</td>
<td>0.27</td>
<td>0.45</td>
</tr><tr>
<td>Basis Sharing</td>
<td>17.09</td>
<td>60.08</td>
<td>0.25</td>
<td>0.65</td>
<td>0.66</td>
<td>0.40</td>
<td>0.31</td>
<td>0.69</td>
<td>0.26</td>
<td>0.46</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>13.68</td>
<td>44.87</td>
<td>0.27</td>
<td>0.68</td>
<td>0.67</td>
<td>0.43</td>
<td>0.33</td>
<td>0.71</td>
<td>0.28</td>
<td>0.48</td>
</tr></table>

exception on PTB at 30%. D-Rank can even achieve a PPL of 80.69, which is about 8 lower than Basis Sharing. As compression tightens from 20% to 50%, all methods' performance degrades, but ours degrades more gracefully, yielding a stronger accuracy-compression trade-off; PTB is the most compression-sensitive among the language modeling datasets.

We also provide the results of D-Rank on LLaMA-3-8B. As shown in Table 4, D-Rank consistently outperforms all baselines under the 20%o compression ratio. Compared with baselines, it achieves notably lower perplexity on WikiText-2 and C4. For example, D-Rank can get the lowest PPL of nearly 45 on C4, which is at least 15 lower than baselines. D-Rank also obtains the best zero-shot accuracies on reasoning tasks such as 71% on PIQA and 67% on WinoGrande. On average,D-Rank delivers the highest overall score of 48%, demonstrating superior performance over baselines.

**Performance on different LLMs.** Table 6 re-ports results on three representative LLMs under a 20% compression ratio. Conventional SVD-based methods suffer from extremely high per-plexities, while SVD-LLM and Basis Sharing provide partial improvements.

<!-- 13 SVD-LLM 12 Basis-Sharing D-Rank (Ours) 11 Taa 10 9 8 7 20% 30% 40% 50% Compression Ratio -->
![](./images/a2ab393c8b31979d.jpg)

In contrast, D-Rank achieves the best overall performance across all models. For instance,on LLaMA-2-7B, D-Rank obtains a PPL of 7.51, outperforming SVD-LLM's PPL of 8.5 and Ba-sis Sharing's PPL of 7.57. Similarly, on Mistal-7B we reach the PPL of 7.41,which is lower than all baselines. These results highlight the Figure 3:LoRA fine-tuning PPL (↓) results of robustness of D-Rank across different LLMs. compressed LLaMA-7B

**Performance** **on** **different** **scales.** Table 7 further evaluates D-Rank on LLaMA models with three scales of 7B, 13B, and 30B. It can be seen that D-Rank consistently achieves the lowest perplexity.On LLaMA-13B,our approach achieves a PPL of 6.30, lower than 6.61 of SVD-LLM and 6.47 of Basis Sharing. On the largest 30B model, D-Rank yields 5.33, which is better than both Basis Sharing's PPL of 5.47 and SVD-LLM's PPL of 5.63. This demonstrates that D-Rank scales effectively to larger models, maintaining superior accuracy under compression.

Table 5: Evaluation of PPL(↓) with differentβ in D-Rank for different grouped layersnon LLaMA-7B under compression ratios from 20% to 50%. S of all tasks is obtained with WikiText-2

<table border="1" ><tr>
<td rowspan="2"></td>
<td rowspan="2"># Compression ratio #Grouped layers</td>
<td colspan="3">20%</td>
<td colspan="3">30%</td>
<td colspan="3">40%</td>
<td colspan="3">50%</td>
</tr><tr>
<td>2</td>
<td>3</td>
<td>4</td>
<td>2</td>
<td>3</td>
<td>4</td>
<td>2</td>
<td>3</td>
<td>4</td>
<td>2</td>
<td>3</td>
<td>4</td>
</tr><tr>
<td></td>
<td>Basis Sharing</td>
<td>7.74</td>
<td>7.72</td>
<td>7.65</td>
<td>9.25</td>
<td>9.27</td>
<td>9.18</td>
<td>12.39</td>
<td>12.60</td>
<td>12.58</td>
<td>19.99</td>
<td>20.06</td>
<td>20.86</td>
</tr><tr>
<td rowspan="6">β</td>
<td>0.2</td>
<td>7.51</td>
<td>7.53</td>
<td>7.40</td>
<td>9.04</td>
<td>9.00</td>
<td>8.93</td>
<td>12.11</td>
<td>12.13</td>
<td>12.08</td>
<td>20.19</td>
<td>19.60</td>
<td>19.72</td>
</tr><tr>
<td>0.25</td>
<td>7.48</td>
<td>7.42</td>
<td>7.36</td>
<td>8.99</td>
<td>8.98</td>
<td>8.91</td>
<td>12.08</td>
<td>12.11</td>
<td>12.06</td>
<td>20.05</td>
<td>19.53</td>
<td>19.65</td>
</tr><tr>
<td>0.3</td>
<td>7.45</td>
<td>7.37</td>
<td>7.37</td>
<td>8.97</td>
<td>8.99</td>
<td>8.87</td>
<td>12.03</td>
<td>12.00</td>
<td>12.04</td>
<td>19.87</td>
<td>19.46</td>
<td>19.49</td>
</tr><tr>
<td>0.35</td>
<td>7.47</td>
<td>7.40</td>
<td>7.35</td>
<td>9.00</td>
<td>8.89</td>
<td>8.90</td>
<td>11.99</td>
<td>11.98</td>
<td>12.02</td>
<td>19.85</td>
<td>19.32</td>
<td>19.41</td>
</tr><tr>
<td>0.4</td>
<td>7.50</td>
<td>7.39</td>
<td>7.35</td>
<td>9.07</td>
<td>8.93</td>
<td>9.02</td>
<td>12.04</td>
<td>12.01</td>
<td>12.07</td>
<td>19.83</td>
<td>19.39</td>
<td>19.35</td>
</tr><tr>
<td>0.45</td>
<td>7.54</td>
<td>7.39</td>
<td>7.36</td>
<td>9.12</td>
<td>8.96</td>
<td>9.04</td>
<td>12.06</td>
<td>12.03</td>
<td>12.10</td>
<td>19.89</td>
<td>19.53</td>
<td>19.46</td>
</tr></table>

**Performance** **under** **LoRA** **fine-tuning.** D-Rank can combine with LoRA fine-tuning to re-cover performance. Our LoRA fine-tuning settings include lora\_=8,laalpa=32,,and learningrat=1-4, and we use default settings for all other hyperparameters in the Hugging Face PEFT. Each compressed model is fine tuned with WikiText-2 training dataset for two epochs. Figure 3 illustrates the LoRA fine-tuning perplexity (PPL) results of LLaMA-7B with 20-50%

<!-- 8 -->

Table 6: PPL (↓) of different LLMs under 20% Table 7: PPL (↓) of LLaMA-7B,13B,30B under compression ratio on WikiText-2 20% compression ratio on WikiText-2

<table border="1" ><tr>
<td>Method</td>
<td>LLaMA-7B</td>
<td>LLaMA-2-7B</td>
<td>Mistral-7B</td>
</tr><tr>
<td>SVD</td>
<td>20061</td>
<td>18192</td>
<td>159627</td>
</tr><tr>
<td>FWSVD</td>
<td>1721</td>
<td>2360</td>
<td>6357</td>
</tr><tr>
<td>ASVD</td>
<td>11.14</td>
<td>10.10</td>
<td>13.72</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.94</td>
<td>8.50</td>
<td>10.21</td>
</tr><tr>
<td>Basis Sharing</td>
<td>7.74</td>
<td>7.70</td>
<td>7.57</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>7.45</td>
<td>7.51</td>
<td>7.41</td>
</tr></table>

<table border="1" ><tr>
<td>Method</td>
<td>7B</td>
<td>13B</td>
<td>30B</td>
</tr><tr>
<td>SVD</td>
<td>20061</td>
<td>946.31</td>
<td>54.11</td>
</tr><tr>
<td>FWSVD</td>
<td>1630</td>
<td>OOM</td>
<td>OOM</td>
</tr><tr>
<td>ASVD</td>
<td>11.14</td>
<td>6.74</td>
<td>22.71</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.94</td>
<td>6.61</td>
<td>5.63</td>
</tr><tr>
<td>Basis Sharing</td>
<td>7.75</td>
<td>6.47</td>
<td>5.47</td>
</tr><tr>
<td>D-Rank (Ours)</td>
<td>7.45</td>
<td>6.30</td>
<td>5.33</td>
</tr></table>

compression using different methods. Across all settings, D-Rank consistently yields lower PPLthan both SVD-LLM and Basis-Sharing. The advantage is alreadyevident at 20% compression, and the gap steadily widens as the compression ratio increases. For instance, when compression reaches 50%, our approach reduces PPL by more than 2 compared to SVD-LLM, highlighting its stronger robustness under aggressive compression. These results demonstrate that D-Rank maintains a more favorable accuracy and compression trade-off than existing baselines.

**Performance with calibration data from different** **datasets.** As shown in Table 8, we use C4 as cal-ibration data to get S to perform compression on LLaMA-7B at a 20% ratio and then evaluate PPL on both C4 and WikiText-2. We observe that while Basis Sharing achieve moderate reductions in PPL compared to SVD-LLM, D-Rank consistently yields the lowest values across different group sizes. For example, when grouping 4 layers,our approach re-duces the PPL on C4 from 11.42 of Basis Sharing to 10.78,and on WikiText-2from 11.08 to 9.78. This demonstrates that D-Rank not only preserves perfor-

Table 8**:** Evaluation of PPL(↓) of LLaMA-7B at 20% compression ratio using C4 as cali-bration data. Evaluation is done on C4 and Wikitext-2

<table border="1" ><tr>
<td>Method</td>
<td>Grouped layers</td>
<td>C4 PPL</td>
<td>Wikitext-2 PPL</td>
</tr><tr>
<td>SVD-LLM</td>
<td></td>
<td>11.84</td>
<td>11.60</td>
</tr><tr>
<td rowspan="4">Basis Sharing</td>
<td>2</td>
<td>11.53</td>
<td>10.90</td>
</tr><tr>
<td>345</td>
<td>11.44</td>
<td>10.98</td>
</tr><tr>
<td>4</td>
<td>11.42</td>
<td>11.08</td>
</tr><tr>
<td>5</td>
<td>11.31</td>
<td>11.16</td>
</tr><tr>
<td rowspan="4">D-Rank (Ours)</td>
<td>2</td>
<td>11.07</td>
<td>9.99</td>
</tr><tr>
<td>3</td>
<td>10.88</td>
<td>10.00</td>
</tr><tr>
<td>4</td>
<td>10.78</td>
<td>9.78</td>
</tr><tr>
<td>5</td>
<td>10.71</td>
<td>9.89</td>
</tr></table>

mance on the Wikitext-2 calibration dataset but also transfers better to out-of-distribution evaluation, highlighting its effectiveness and robustness.

<!-- Dense SVD-LLM 2000 Basis Sharing D-Rank(Ours) 2as/SuaxoL 1500 1000 500 0 20% 30% 40% 50% Compression Ratio -->
![](./images/72edea3cedb6cf8e.jpg)

**Choice** **of** **the** β. Table 5 studies the effect of redis-tributing ranks among theWQ,WKandWVma-trices, where the adjustment ratio is denoted by β.We evaluate LLaMA-7B under different compression ratios from 20% to 50% on WikiText-2. The results indicate that an appropriate choice ofβ significantly improves performance compared with the Basis Shar-ing baseline. In particular,β=0.3-0.4 consistently yields the lowest PPL across different settings. For example, at 30% compression, D-Rank achieves PPL of 8.87 when group size is 4 compared to PPL of 9.18for Basis Sharing; at 40% compression,β=0.35 Figure 4: Throughput of dense LLaMA-7B gives a PPL of 11.98, clearly better than 12.58 from model and the compressed model with Basis Basis Sharing. These results show that shifting part Sharing baseline and D-Rank under of the rank budget fromWQ,WKoWVhelps the compression ratios from 20% to 50%.

model preserve more informative representations, and that a moderate redistribution ofβ around 0.3-0.4 is most effective.

**Hardware** **performance** **of** **throughput.** Figure 4 reports the throughput of LLaMA-7B under different compression ratios ranging from 20% to 50%. As shown in the figure,all compressed models surpass the dense baseline in terms of tokens processed per second, and the improvement becomes more pronounced as the compression ratio increases. Notably, D-Rank consistently achieves the highest throughput among all approaches. For instance, at 50% compression, our approach reaches nearly 2,200 tokens/sec,which exceeds both SVD-LLM and Basis Sharing and offers more than a 60% gain over the dense model. These results confirm1 that D-Rank not only preserves accuracy but also brings substantial acceleration benefits in real inference scenarios.

## 5 CONCLUSION

In this paper, we present **D-Rank,** a novel SVD-based compression framework for large language models. Unlike conventional SVD-based methods, D-Rank dynamically allocates retained ranks for weight matrices across layers to preserve critical information by introducing a novel metric called effective rank to measure weight matrices' information density. By jointly balancing rank distribution across attention layers according to the effective rank ofWQ,WK,WV,our method achieves a better compression performance. Extensive experiments on different architectures and scales demonstrate that D-Rank consistently reduces perplexity and improves zero-shot reasoning accuracy under 20-50% compression. Moreover, D-Rank remains robust across random seeds and can be seamlessly combined with LoRA fine-tuning to further enhance performance. Overall, D-Rank establishes a practical and effective approach for deploying compression on LLMs.

<!-- 9 -->

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist,and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024.

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2357-2367,Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Fluctuation-based adaptive struc-tured pruning for large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp.10865-10873,2024.

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and Jamnes Hensman. SliceGPT: Compress large language models by deleting rows and columns. In The Twelfth International Conference on Learning Representations, 2024a.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron,Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213-100240,2024b.

Haolei Bai, Siyong Jian, Tuo Liang, Yu Yin, and Huan Wang. Ressvd: Residual compensated svd for large language model compression. arXiv preprint arXiv:2505.20112,2025.

Pratyay Banerjee, Kuntal Kumar Pal, Arindam Mitra, and Chitta Baral. Careful selection of knowledge to solve open book question answering. In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp.6120-6129,Florence,Italy, July 2019. Association for Computational Linguistics.

Klaudia Bałazy, Mohammadreza Banaei, Karl Aberer, and Jacek Tabor. Lora-xs: Low-rank adaptation with extremely small number of parameters,2025.URL https://arxiv.org/abs/2405. 17604.

Matan Ben Noach and Yoav Goldberg. Compressing pre-trained language models by matrix de-composition. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing (AACL-IJCNLP), pp. 884-889,Suzhou, China, 2020. Association for Computational Linguistics.

Srinadh Bhojanapalli, Ayan Chakrabarti, Andreas Veit, Michal Lukasik, Himanshu Jain, Frederick Liu, Yin-Wen Chang, and Sanjiv Kumar. Leveraging redundancy in attention with reuse transformers, 2022.URL https://openreview.net/forum?id=V37YFdfFgN.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In AAAI Conference on Artificial Intelligence, 2019.

Jerry Chee,Yaohui Cai,Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. Advances in Neural Information Processing Systems, 36: 4396-4429,2023.

<!-- 10 -->

Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang,and Ping Luo. EfficientQAT: Efficient quantization-aware training for large language models, 2025. URL https: //openreview.net/forum?id=6Mdvq0bPyG.

Tianlong Chen, Yu Cheng, Zhe Gan, Lu Yuan, and Zhangyang Zhang. The lottery ticket hypothesis for pre-trained bert networks. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457,2018. URL https://api.semanticscholar.org/CorpusID: 3922816.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transfoormers. In International Conference on Learning Representations, 2019.

Emily Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, and Rob Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, NIPS'14, pp. 1269-1277,Cambridge, MA, USA,2014.MIT Press.

Flavio Di Palo, Prateek Singhi, and Bilal Fadlallah. Performance-guided LLM knowledge distillation for efficient text classification at scale. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 9311-9328. Association for Computational Linguistics, January 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv-2407,2024.

Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni,and Emma Strubell. Energy considerations of large language model inference and efficiency optimizations. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 32556-32569, Vienna, Austria, July 2025. Association for Computational Linguistics.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning, pp. 10323-10337. PMLR, 2023.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323,2022.

Shangqian Gao, Ting Hua, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. Adaptive rank selections for low-rank approximation of language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 227-241,2024.

Gene H Golub, Alan Hoffman, and Gilbert W Stewart. A generalization of the eckart-young-mirsky matrix approximation theorem. Linear Algebra and its applications, 88:317-327,1987.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024.

Tamir David Hay and Lior Wolf. Dynamic layer tying for parameter-efficient transformers. In The Twelfth International Conference on Learning Representations, 2024.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS Deep Learning and Representation Learning Workshop, 2015.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. Language model compression with weighted low-rank factorization. In International Conference on Learning Representations (ICLR), 2022.

<!-- 11 -->

Dou Hu,Lingwei Wei, Wei Zhou, and Songlin Hu. An information-theoretic multi-task representation learning framework for natural language understanding. InProceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp.17276-17286,2025.

Ting Hua, Yen-Chang Hsu, Felicity Wang, Qian Lou, Yilin Shen, and Hongxia Jin. Numerical optimizations for weighted low-rank estimation on language models. In Proceedingsof the 2022Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1404-1416, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford,Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,Thomas Wang,Timothée Lacroix, and William El Sayed. Mistral 7b, 2023a. URL https://arxiv. org/abs/2310.06825.

Yuxin Jiang,Chunkit Chan,Mingyang Chen, and Wei Wang. Lion: Adversarial distillation of proprietary large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp.3134-3154,Singapore, December 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.189.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4163-4174,2020.

Qingyuan Li, Ran Meng, Yiduo Li, Bo Zhang, Liang Li, Yifan Lu, Xiangxiang Chu,Yerui Sun, and Yuchen Xie. A speed odyssey for deployable quantization of llms. arXiv preprint arXiv:2311.09550,2023.

Zhiteng Li,Xianglong Yan, Tianao Zhang,HHaotong Qin, Dong Xie, Jiang Tian, zhongchao shi, Linghe Kong, Yulun Zhang, and Xiaokang Yang. ARB-LLM: Alternating refined binarizations for large language models. In The Thirteenth International Conference on Learning Representations, 2025.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang,Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao,Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems,6: 87-100,2024.

Gui Ling, Ziyang Wang, and Qingwen Liu. SSlimgpt: Layer-wise structured pruning for large language models. Advances in Neural Information Processing Systems, 37:107112-107137,2024.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. LLM-QAT: Data-free quantization aware training for large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024,pp.467-484,Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.26.

Zequan Liu, Jiawen Lyn, Wei Zhu, Xing Tian, and Yvette Graham. ALoRA: Allocating low-rank adaptation for fine-tunin1g large language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 622-641,Mexico City, Mexico, June 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.35.

Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language model. Journal of machine learning research,24(253):1-15, 2023.

Sasha Luccioni, Bruna Trevelin, and Margaret Mitchell. The environmental impacts of ai-primer. Hugging Face Blog, 2024.

<!-- 12 -->

Xinyin Ma,Goongfan Fang, and Xinchao Wang. Llm-pruner: On the structural pruning of large language models. Advances in neural information processing systems, 36:21702-21720,2023.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small language models to reason. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1773-1781, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-short.151.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313-330,1993.

Fanxu Meng,Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems, 37: 121038-121072,2024.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.

Richard Petri, Grace Li Zhang, Yiran Chen, Ulf Schlichtmann, and Bing Li. Powerpruning: Selecting weights and activations for power-efficient neural network acceleration. In 2023 60th ACM/IEEE Design Automation Conference (DAC), pp.1-6.IEEE,2023.

Wang Qinsi, Jinghan Ke, Masayoshi Tomizuka, Kurt Keutzer,and Chenfeng Xu. Dobi-SVD: Differ-entiable SVD for LLM compression and some new perspectives. In The Thirteenth International Conference on Learning Representations, 2025.

Ruidi Qiu, Amro Eldebiky, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Ulf Schlichtmann, and Bing Li.Oplixnet: Towards area-efficient optical split-comnplex networks with real-to-complex data assignment and knowledge distillation. In 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), pp.1-6.IEEE,2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J.Mach.Learn. Res., 21(1),January 2020.

Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, and Andrey Kuznetsov. The shape of learning: Anisotropy and intrinsic dimensions in transformer-based models. arXiv preprint arXiv:2311.05928,2023.

Machel Reid, Edison Marrese-Taylor, and Yutaka Matsuo. Subformer: Exploring weight sharing for parameter efficiency in generative transformers, 2021. URL https://arxiv.org/abs/ 2101.00234.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversar-ial winograd schema challenge at scale. Commun. ACM, 64(9):99-106, August 2021.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, 2024.

Wenhao Sun, Grace Li Zhang, Huaxi Gu, Bing Lil, and Ulf Schlichtmann. Class-based quantization for neural networks. In 2023 Design, Automation & Test in Europe Conference & Exhibition (DATE), pp.1-6.IEEE,2023.

Lintang Sutawika, Hailey Schoelkopf, Leo Gao, Baber Abbasi, Stella Biderman, Jonathan Tow, ben fattori, Charles Lovering, farzanehnakhaee70, Jason Phang, Anish Thite, Fazz, Aflah, Niklas Muennighoff, Thomas Wang, sdtblck, nopperl, gakada, tttyuntian,researcher2, Julen Etxaniz,Chris, Hanwool Albert Lee, Zdeněk Kasner, Khalid, LSinev, Jeffrey Hsu, Anjor Kanekar, KonradISzafer, and AndyZwei. Eleutherai/lm-evaluation-harness: v0.4.3,July 2024.

<!-- 13 -->

Sho Takase and Shun Kiyono. Lessons on parameter sharing across layers in transformers. In Nafise Sadat Moosavi, Iryna Gurevych,Yufang Hou, Gyuwan Kim, Young Jin Kim, Tal Schuster, and Ameeta Agrawal (eds.), Proceedings of the Fourth Workshop on Simnple and Efficient Natural Language Processing (SustaiNLP), pp. 78-90, Toronto, Canada (Hybrid), July 2023. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971,2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cris-tian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux,Thibaut Lavril,Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang,Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023b. URL https://arxiv.org/abs/2307.09288.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Anna Korho-nen, David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5797-5808, Florence, Italy, July 2019.Association for Computational Linguistics. doi: 10.18653/v1/P19-1580.

Jingcun Wang,Yu-Guang Chen, Ing-Chao Lin, Bing Li, and Grace Li Zhang. Basis sharing: Cross-layer parameter sharing for large language model compression. In The Thirteenth International Conference on Learning Representations, 2025a.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. SVD-LLM: Truncation-aware singular value decomposition for large language model compression. In The Thirteenth International Conference on Learning Representations,2025b.

Yuxin Wang, Minghua Ma, Zekun Wang, Jingchang Chen, Huiming Fan,Liping Shan, Qing Yang, Dongliang Xu, Ming Liu, and Bing Qin. Cfsp: An efficient structured pruning framework for LLMs with coarse-to-fine activation information. In Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), pp. 9311-9328. Association for Computational Linguistics, January 2025c.

Lai Wei, Zhiquan Tan, Chenghai Li, Jindong Wang, and Weiran Huang. Diff-erank: A novel rank-based metric for evaluating large language models. Advances in Neural Information Processing Systems, 37:39501-39521,2024.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International conference on machine learning, pp. 38087-38099. PMLR, 20)23.

Tong Xiao, Yinqiao Li, Jingbo Zhu,Zhengtao Yu, and Tongran Liu. Sharing attention weights for fast transformer. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pp. 5292-5298. International Joint Conferences on Artificial Intelligence Organization,2019.

Xinhao Yao, Hongjin Qian, Xiaolin Hu, Gengze Xu, Wei Liu, Jian Luan, Bin Wang,and Yong Liu. Theoretical insights into fine-tuning attention mechanism: Generalization and optimization. arXiv preprint arXiv:2410.02247,2024.

<!-- 14 -->

Zhihang Yuan,Yuzhang Shang, Yue Song, Dawei Yang,QiangWu,Yan Yan,and Guangyu Sun. ASVD: Activation-aware singular value decomposition for compressing large language models, 2025.URL https://openreview.net/forum?id=HyPofygOCT.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Annual Meeting of the Association for Computational Linguistics, 2019.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adalora: Adaptive budget allocation for parameter-efficient fine-tuning. In International Conference on Learning Representations (ICLR), 2023.

Yingtao Zhang,Haoli Bai, Haokun Lin, Jialin Zhao, Lu Hou, and Carlo Vittorio Cannistraci. Plug-and-play: An efficient post-training pruning method for large language models. In The Twelfth International Conference on Learning Representations, 2024.

Weibo Zhao, Yubin Shi, Xinyu Lyu, Wanchen Sui, Shen Li, and Yong Li. Aser: activation smoothing and error reconstruction for large language model quantization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp.22822-22830,2025.

Yilong Zhao,Chien-Yu Lin,Kan Zhu, Zihao Ye,Lequn Chen,Size Zheng,Luis Ceze,Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. Proceedings of Machine Learning and Systems, 6:196-209,2024.

Qihuang Zhong, Liang Ding, Li Shen, Juhua Liu, Bo Du, and Dacheng Tao. Revisiting knowledge distillation for autoregressive language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10900-10913, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.acl-long.587.

<!-- 15 -->

## A APPENDIX

### A.1 RELATED WORK

**Large Language** **Model (LLM)** **Compression.** LLMs typically contain billions of parameters, making traditional training-based compression techniques impractical due to the high computational cost. To alleviate this, post-training compression methods have been widely explored, mainly falling into three major categories: knowledge distillation, pruning, and quantization. Knowledge distillation (KD) (Hinton et al., 2015; Jiao et al., 2020) compresses LLMs by training a smaller student model to mimic the behavior of a larger teacher model. The student learns from the teacher's logits or intermediate representations, thereby reducing the parameter count and inference cost while aiming to preserve performance. However, recent studies (Zhong et al., 2024; Di Palo et al., 2025;Agarwal et al., 2024) have shown that student models often exhibit limited generalization capability compared to their teachers. Pruining removes redundant weights or channels from the original model to produce a sparse subnetwork (Ashkboos et al., 2024a; Sun et al., 2024; Zhang et al., 2024). Unstructured pruning method sets individual weights to zero (Frantar & Alistarh, 2023), while structured pruning removes entire channels or attention heads (An et al., 2024;Wang et al., 2025c; Ling et al., 2024). Although pruning reduces memory and computation, many pruning schemes require retraining,second-order information, or manual sparsity tuning, and theyoften suffer from performance degradation especially at high sparsity levels (Chen et al., 2021). Quantization reduces model size by representing weights and activations with lower-bit precision such as 8-bit, 4-bit, or even 1-2 bits (Frantar et al., 2022; Zhao et al., 2025). This significantly lowers memory usage and enables faster inference. However, aggressive low-bit quantization (such as 1-2 bits) can introduce substantial accuracy drops (Chee et al., 2023; Li et al., 2025), and quantization-aware training (QAT) requires large datasets and heavy computation (Chen et al., 2025; Liu et al., 2024a), limiting its practicality.

**SVD-based** **LLM** **Compression.** Singular Value Decomposition (SVD) reduces matrix dimen-sionality by truncating the smallest singular values and factorizing the original matrix into three smaller low-rank matrices that approximate it (Golub et al., 1987). SVD-based compression for large language models (LLMs) can simultaneously preserve semantic information and reduce the number of parameters, while allowing the accuracy drop to be controlled. Early studies such as (Denton et al., 2014) demonstrated that applying SVD to convolutional neural networks (CNNs) can substantially accelerate inference without sacrificing accuracy. Building on this idea, (Ben Noach & Goldberg, 2020) applied truncated SVD to BERT-base to obtain an optimal low-rank approximation,which provided high-quality initialization for fine-tuning. However, conventional SVD-based compression assumes all parameters are equally important (Hua et al., 2022), and typically requires fine-tuning after compression to recover performance. To address this limitation, (Hsu et al., 2022) proposed the FWSVD method,wwhich integrates Fisher information into the low-rank decomposition objective to better align the decomposition with task-specific loss. Yet, FWSVD only considers weight impor-tance and overlooks activation outliers or distributional shifts. To mitigate this, (Yuan et al., 2025) introduced ASVD, which preprocesses weights using activation distributions and incorporates outlier influence before performing SVD. Nevertheless, ASVD does not update model parameters after truncation. More recently, (Wang et al., 2025b) presented SVD-LLM, which improves compression efficiency by employing truncation-aware data whitening to align singular values with compression loss and introducing layer-wise closed-form updates. Moreover, Dobi-SVD (Qinsi et al., 2025) intro-duces a differentiable truncation mechanism combined with theoretical analysis and a weight update formulation, which significantly improves performance under high compression ratios. ResSVD (Bai et al., 2025) leverages the residual matrix generated during the SVD truncation process to reduce truncation errors, and compresses only the latter layers of the model to avoid error accumulation. Despite these advances, most existing studies such as (Yuan et al., 2025; Wang et al., 2025b) focus on compressing and recovering individual layers of large language models, or rely on memory-intensive techniques such as training or backpropagation (Qinsi et al., 2025). However, little work has explored the compressibility relationships across different layers, and the variation in compressibility among different layer groups remains largely underexplored.

**Parameter** **Sharing.** Model compression through parameter sharing achieves size reduction by reutilizing weight matrices across multiple layers. (Dehghani et al., 2019) proposed the Universal Transformer, all layers share the same set of parameters, akin to the RNNs, leading to signifi-cant parameter reduction. (Reid et al., 2021) categorizes the parameters into attention-related and feedforward-related groups for transformer-based models. These parameters are shared within their respective groups, thereby reducing the overall parameters count while retaining model adaptability. Selective weight sharing is applied to a subset of layers by (Takase & Kiyono,2023),rather than across all layers. Unlike traditional weight sharing, (Xiao et al., 2019); (Bhojanapalli et al.,2022) explores sharing attention scores across layers. It crucially reduces computational and memory overhead.(Hay & Wolf, 2024) introduce a novel framework, named Dynamic Tying, where reinforce-ment learning is used to automatically identify optimal layer-wise parameter sharing patterns during training.

<!-- 16 -->

#### A.2 PERFORMANCE WITH DIFFERENT SEEDS TO SELECT THE CALIBRATION DATA FOR COMPRESSION.

Figure 5 compares the perplexity of different methods on LLaMA-7B when using WikiText-2 as calibration data under varying random seeds. We observe that the performance of both SVD-LLM and Basis Sharing fluctuates with the choice of seed, while D-Rank consistentlyachieves lower PPL across all settings. For instance, at seed 13, D-Rank obtains 7.45 compared to 7.9 for SVD-LLM and 7.7 for Basis Sharing, and this advantage remains evident even at larger seeds such as 512 and 1024.These results demonstrate that our approach is not only superior in average performance but also more robust to randomness in calibration data selection.

<!-- 8.2 SVD-LLM Basis-Sharing 8.0 D-Rank (Ours) 7.8 Tad 7.6 7.4 7.2 7.0 13 42 512 1024 Seed -->
![](./images/5bf51537ae2abbaf.jpg)

Figure 5: Comparison of PPL with baselines onLLaMA-7Bmodel when selecting the calibration data from Wikitext-2 with different seeds to compute S

### A.3 RANK ALLOCATION VIA LAGRANGE MULTIPLIERS

Letkgbe the retained rank for groupg∈{1,⋯,G},Rff(g)the effective rank (information measure), w the parameter cost per unit rank for group g, and Tbudgetthe total rank cost budget as defined in Section 3.2. We minimize the loss under a budget constraint:

$\min _{k_{1},\cdots ,k_{G}}\ell _{\text {total}}=\sum _{g=1}^{G}\frac {\mathcal {R}_{\mathrm {eff}}(g)}{k_{g}}$  s.t. $\sum _{g=1}^{G}k_{g}ω=\mathcal {T}_{\text {budget}}$  (13)

The Lagrangian is

$$\mathcal {F}\left(\left\{k_{g}\right\},λ\right)=\sum _{g=1}^{G}\frac {\mathcal {R}_{\text {eff}}(g)}{k_{g}}+λ\left(\sum _{g=1}^{G}k_{g}ω-\mathcal {T}_{\text {budget}}\right)\tag{14}$$

Setting the derivative w.r.t. eachkto zzero:

$$\frac {\partial \mathcal {F}}{\partial k_{g}}=-\frac {\mathcal {R}_{\mathrm {eff}}(g)}{k_{g}^{2}}+λω=0Longrightarrowk_{g}=\sqrt {\frac {\mathcal {R}_{\mathrm {eff}}(g)}{λω}}\tag{15}$$

<!-- 17 -->

Hence the optimal ranks follow the proportionality

$$k_{g}\propto \frac {\sqrt {\mathcal {R}_{\mathrm {eff}}(g)}}{\sqrt {ω}}\tag{16}$$

Let C be the proportionality constant. Using the budget constraint,

$$\sum _{g=1}^{G}k_{g}ω=\sum _{g=1}^{G}\left(C\frac {\sqrt {\mathcal {R}_{\mathrm {eff}}(g)}}{\sqrt {ω}}\right)ω=C\sum _{g=1}^{G}\sqrt {\mathcal {R}_{\mathrm {eff}}(g)ω}=\mathcal {T}_{\text {budget}}\tag{17}$$

so we have:

$$C=\frac {\mathcal {T}_{\text {budget}}}{\sum _{j=1}^{G}\sqrt {\mathcal {R}_{\mathrm {eff}}(j)ω}}\tag{18}$$

Substituting C back yields the final closed-form allocation:

$$k_{g}=\frac {\mathcal {T}_{\text {budget}}}{\sum _{j=1}^{G}\sqrt {\mathcal {R}_{\text {eff}}(j)ω}}·\frac {\sqrt {\mathcal {R}_{\text {eff}}(g)}}{\sqrt {ω}}\tag{19}$$

Interpretation. Equation 16-19 show that groups with larger information contentReff(g)receive higher ranks, whereas groups with higher parameter cost w receive fewer ranks, all under the fixed budgetTbudgett·

<!-- 18 -->

