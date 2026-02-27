# QSVD: Efficient Low-rank Approximationfor Unified

# Query-Key-Value Weight Compression in

# Low-Precision Vision-Language Models

**Yutong Wang¹* Haiyu Wang¹*Sai Qian Zhang1,2**

1Tandon School of Engineering, New York University

2Courant InstituteofMathematical Sciences, New York University

{yw6594,hw3689, sai.zhang}@nyu.edu

## Abstract

Vision-Language Models (VLMs) are integral to taskssuch as image captioning and visual question answering, but their high computational cost, driven by large memory footprints and processing time, limits their scalability and real-time ap-plicability. In this work, we propoose leveraging Singular-Value Decomposition (SVD) over the joint query (Q), key (K), and value (V) weight matrices to reduce KV cache size and computational overhead. We in addition introduce an efficient rank allocation strategy that dynamically adjusts the SVD rank based on its impact on VLM accuracy, achieving a significant reduction in both memory usage and computational cost. Finally, we extend this approach by applying quantization to both VLM weights and activations, resulting in a highly efficient VLM. Our method outperforms previous approaches that rely solely on quantization or SVD by achieving more than 10% accuracy improvement while consuming less hardware cost, making it better for real-time deployment on resource-constrained devices.We open source our code at https://github.com/SAI-Lab-NYU/QSVD.

## 1 Introduction

Vision-Language Models (VLMs) are crucial for advancing artificial intelligence by bridging the gap between visual perception and natural language understanding. By enabling machines to interpret and generate both visua1 and textual information, VLMs open up a wide range of applications, such as image captioning [63,17,8,11], visual question answering [7,3,47],and content-based search [22, 43]. These models are vital for tasks where visual context is needed to fully understand textual queries or vice versa, such as healthcare [34,3,18], education [61], and interactive entertainment [44,26].

Despite their strong performance, Vision-Language Models (VLMs) incur substantial computational costs due to the intensive processing required to integrate high-dimensional visual and textual data. Additionally, their autoregressive token generation places significant pressure on memory bandwidth, becoming a major bottleneck for inference speed. To enable practical deployment in latency-sensitive and resource-constrained environments, it is essential to reduce both the computational overhead and the size of the Key-Value (KV) cache, without compromising model accuracy.

To address this issue, particularly the high memory usage introduced by Multi-Head Attention (MHA), several variants have been proposed, such as Grouped-Query Attention [1] and Multi-Query Attention [42, 1], which aim to reduce the number of KV projections while maintaining performance. A recent proposal, Multi-Head Latent Attention (MLA) in the DeepSeek-v3 model [31], offers a novel approach to improving VLM efficiency. It significantly reduces the KV cache size by compressing the KV cache into a latent vector, thereby enhancing inference efficiency.

*Authors contributed equally; the order of authorship was assigned randomly.

<!-- 39th Conference on Neural Information Processing Systems (NeurIPS 2025). -->

<!-- S7071o08I [T.so] IAz6C9T.OISC:AIXTe -->

<!-- “An image of two golden retrievers" Language model Projection Embedding Visual Encoder KXe1 "Describe the image" -->
![](./images/4ec015881b1abdba.jpg)

**(a)**

<!-- Calibration . dataset Step 1: Joint SVD over QKV Weight QSVD Q:What is the customer review Step 3: 女eL Post-training Quantization Q:What is the brand of the car Image Step 2:Adaptive rank allocation for SVD Outlier smoothing Embedding Visual encoder = x r 3.91.24.6-1 Projection 3-18.61. n Quantization mage Language model Loss Token Concatenation (b) -->
![](./images/7baa0de94a787bf5.jpg)

Figure 1: (a) An example on vision-language model. (b) An overview of QSVD.

Building on the insights from MLA, this work proposes the application of Singular-Value Decompo-sition (SVD), which has proven effective in reducing both KV cache size and compoutational cost in prior research [51,650,52,28,25,5,49],to the joint weight matrices of the query,key,and value. This approach significantlyreduces the KV cache size by storing only the latent vectors instead of separate key and value vectors. Additionally, we introduce a novel rank allocation scheme, which investigates the importance of each singular value in relation to VLM accuracy. This results in a minimal SVD rank with the minimal impact on the model accuracy. Finally, we extend this approach by applying quantization to both VLM weights and activations. The proposed framework, termed QSVD, results in an extremely efficient VLM that outperforms previous methods that relied solely on quantization or SVD. Specifically, our contribution can be summarized as follows:

·QSVD proposes applyingsingular value decomposition to the combined weight matrices of the query, key, and value projections. This technique substantially reduces the size of the KV cache, comnputational overhead, and weight storage, resulting in significant improvements in hardware efficiency.

·To improve the accuracy of SVD-based compression in VLMs, QSVD proposes a novel importance scoring method that quantifies each singular value's contribution to overall model performance, allowing for rank-based truncation that minimizes accuracy degradation.

·Quantization is applied alongside SVD decomposition to both the weights and activations of the VLM. We propose an efficient method to eliminate outliers under the SVD framework, enabling low-precision operation that reduces the memory footprint of both the KV cache and model weights, while incurring minimal impact on accuracy.

## 2 Background and Related Work

### 2.1 Vision Language Model

Vision-Language Models (VLMs) [24,23,33,4,13,48] extend the capabilities of Large Language Models (LLMs) by incorporating visual inputs alongside text, enabling tasks such as visual question answering (VQA) and image captioning. Models such as BLIP and InstructBLIP [24,23] employ data filtering and visual instruction tuning to better align model outputs with human preferences in zero-shot settings. A commonly used architecture, illustrated in Figure 1 (a), processes input images through a visual encoder to produce visual tokens, which are then concatenated with text tokens and passed to a language model for response generation. This concatenation-based design is adopted by widely used models including the LLaVA series [33], SmolVLM [39], PaLI-Gemma [4],and Qwen-VL [48]. Although VLMs demonstrate impressive capabilities, their large size presents challenges in terms of computational efficiency and deployment, particularly on resource-constrained devices. This has led to the development of lightweight alternatives. TinyGPT-V [59] and TinyLLaVA [62] explore efficient designs at smaller scales, while SmolVLM [39] introduces a family of compact models (500M and 2B parameters) that maintain strong performance with much reduced hardware cost.

<!-- 2 -->

<!-- **2.2 Singular Value Decomposition for Large Models** -->

Singular Value Decomposition (SVD) [19] is a widely used matrix factorization technique that decomposes a matrix W∈Rm×into three components: W=UΣVT,where U and V are orthogonal matrices containing the left and right singular vectors of W, and Σ is a diagonal matrix of non-negative singular values arranged in descending order. By retaining only the top r singular values and corresponding vectors, wve obtain arank-r approximation:

$$W\approx U_{r}Σ_{r}V_{r}^{T}\tag{1}$$

withUr∈Rm×r,Σr∈Rr×r,andVr∈Rn×r. Equivalently, the approximation can be expressed asW≈ABby defining $A=U_{r}Σ_{r}^{\frac {1}{2}}$ and $B=Σ_{r}^{\frac {1}{2}}V_{r}^{T}$ . Such low-rank factorizations preserve the most salient structure of W while reducing its dimensionality, enabling matrix compression and accelerating downstream computations.

SVD has been extensively studied as a compression method for LLMs [51,60,52,28,25,5,49].Early efforts [40] directly applies standard SVD to weight matrices but encountered significant compression errors. To address this, FWSVD [16] prioritizes parameters based on Fisher information [37],while ASVD [60] incorporates activation outliers into the factorization. SVD-LLM [52] further reduces compression loss by explicitly minimizing the contribution of each truncated singular value. Most of these methods focus on compressing model weights. In contrast, Palu [6] and [58] have shifted attention to compressing the KV/-Cache, leveraging SVD and low-rank projections to reduce memory footprint. Recent advances include AdaSVD [28], which adaptively compensates for truncation errors and dynamically allocates compression rates according to layer importance, and SVD-LLM V2 [51], which further optimizes singular value truncation via theoretical loss estimation.

Recently, DeepSeek introduces Multi-Head Latent Attention [31], a novel mechanism that integrates low-rank projections directly into the attention computation. Instead of computing attention over the full key and value matrices, this approach projects them into a lower-dimensional latent space using learned projection matrices, effectively reducing the computational and memory costs of multi-head attention without significantly impacting model performance. This latent factorization can be viewed as an implicit low-rank approximation applied dynamically during inference, offering complementary benefits to static weight compression methods such as SVD.

### 2.3 Quantization for Large Models

Post-training quantization (PTQ) has become one of the most used approaches for enabling efficient inference of large models [38,55,2,29,12,41,30,57,45,53,20,9]. For example, AffineQuant [38] replaces the traditional scaling factor with a learned affine transformation to better align weight with the quantization grid.

Another line of work focuses on smoothing outliers in activation distributions, which have shown that activations in LLMs contain severe outliers at the per-channel level [2,9,29,35,30,55],resulting in substantial quantization errors during activation quantization. To address this issue, SmoothQuant [55] reduces activation outliers by shifting part of the activation outliers into the weights, promoting more balanced quantization. Building upon these ideas, techniques like QuaRot [2], DuQuant [29],and SpinQuant [35] incorporate orthogonal transformations to further enhance quantization performance. These transformations maintain computational invariance by preserving the model's output while effectively suppressing outliers. Specifically, let W and X represent the weight and activation matrices, respectively, where X exhibits channelwise outliers,and letY=XWdenote the output. To eliminate outliers in X, an orthogonal matrix H is introduced, satisfying\top=I\top=I This transformation yields an equivalent formulationY=XW=X′W′,whereW′=H\topW andX′=XH. To minimize runtime overhead,W′=H\topWcan be precomputed offline,and X′=XHcan be efficiently integrated into the weight matrices of the previous layer, incurring no additional computational cost. The resulting transformed activation X′ exhibits a smoother distribution with significantly fewer outliers, thus lowering the quantization errors. In parallel, methods such as GPTQ [12], OmniQuant [41], and AWQ [30] focus on optimizing scaling factors and channel-wise equalization during the calibration process.

In the realm of VLMs, quantization presents unique challenges due to the integration of visual and textual modalities. QSLAW [56] introduces a quantization-aware scale learning method with a multimodal warmup strategy that progressively incorporates linguistic and multimodal samples

<!-- 3 -->

<!-- Wq $\square {W_{q}}\approx E\left[\begin{array}{c}{W_{q}^{d}}\\ {}\end{array}\right.Xr\begin{array}{c}E\\ {W_{q}^{u}}\\ \end{array}$ $\begin{array}{c|c|c|c|c|c|c}W_{k}&W_{k}&\approx E&\frac {r}{W_{k}^{d}}&\times r&\frac {E}{W_{k}^{u}}&\approx &\frac {r}{W^{d}}&\times r&\frac {r}{Σ}&\times r\end{array}$ Wv Wv≈E[\;Wvd] $x_{r}\left[\begin{array}{l}\mathrm {\;W}_{v}^{u}\\ hline\end{array}\right.$ -->
![](./images/4e5a2d9c774bb262.jpg)

<!-- 3E E Wq Wk Wv Wconcat 3E Wu 3E ≈E[\begin{array}l\;Wqkvd\end{array}] + Wqu Wku Wvu -->
![](./images/0e2c6ef2b3133af5.jpg)

<!-- E E E E Wk 0 Ck X x E E E Wv Cv -->
![](./images/5eb521cae4193e38.jpg)

<!-- r E Wkd =L Ck L X x E |Wvd|=L Cv -->
![](./images/1988b8a53099fd6a.jpg)

<!-- L X $\times \mathrm {E}\left[\begin{array}{c}\mathrm {\;W}_{\mathrm {Qkv}}^{\mathrm {d}}\\ \mathrm {\;K}_{\mathrm {Qkv}}\end{array}\right]=\mathrm {L}\left[\begin{array}{c}\mathrm {C}_{\mathrm {Qkv}}\end{array}\right.$ -->
![](./images/11f410e2b67bc505.jpg)

(a) (b) (c) (d) (e) (f)

Figure 2: Efficiency analysis of different SVD schemes. (a)(b) are original Q/K/V matrix applied SVD. (c)(d)(e)(f) are proposed concatenated QKV SVD and their corresponding decoding process.

to stabilize training. It also emphasizes group-wise scaling to better handle activation outliers. Q-VLM [46] addresses cross-layer dependency in quantization by leveraging activation entropy as a proxy to guide block partitioning. It formulates a quantization strategy that balances discretization error and search cost, and further optimizes the visual encoder to disentangle cross-layer interactions, enabling more efficient calibration. MBQ [27] proposes a modality-balanced quantization approach that accounts for the distinct gradient distributions of visual and textual tokens during calibration. It applies a modality-aware loss to improve the accuracy of scaling factor estimation. However,to the best of our knowledge, no prior work has combined quantization with SVD for efficient VLM processing in the manner proposed by QSVD.

## 3 Methodology

An overview of QSVD is shown in Figure 1 (b), comprising three key components: joint SVD over the combined QKV weights (Section 3.1), adaptive singular value truncation (Section 3.2),and PTQ over low-rank VLMs (Section 3.3).

### 3.1 Singular-Value Decomposition over Joint QKV Weights

We introduce an efficient SVD-based approach to reduce computation within the multi-head self-attention bolock, as illustrated in Figure 2, where each subfigure denotes: (a) Original QKV matrix in a vision-language model (VLM) without SVD. (b) Applying SVD separately to the weight matrices ofQ,K,and V,where each of Wq,WkandWvis factorized into a down- and up-projection pair. (c) Our proposed approach: concatenating QKV weights before applying SVD. (d) Standard KV computation during prefilling: the input X is multiplied by Wkand $\dot {W}_{v}$ ,and the resultingCkand Cvare stored in memory. (e)Computation with per-matrix SVD: during prefilling, X must be read from memory and multiplied with the down-projection matricesWkandWvto generate low-rank representation of K and V (f) Storage and computation in QSVD: sinceWq,Wk,,andWvshare a common down-projection matrixWqkv,Xis multiplied byWqkvonce to produce the intermediate Cqkv, which is stored and later used to reconstruct the KVectors. Let α, η, and γ denote the weight parameter size, KV cache size, and the computational cost in FLOPs of QKV multiplication, respectively. In the original design, assuming a single-head attention module for simplicity,the combined weight matricesWqWkandWvcollectively coontain a total ofαfp=3E2parameters, where E represents the embedding dimension (Figure 2 (a)).The corresponding KV cache requires a memory footprint ofηfp=L,,where L is the input sequence length (Figure 2 (d)). The total computational cost in FLOPs for generating the key, query, and value vectors isγfp=3LE,where three matrix multiplications are required, each with a size of(LE)and(E×E)

QSVD adopts a more efficient strategy that reduces the number of weight parameters,KV cache size, and overall computational cost. Specifically,the weight matricesWQ,WK,andW,each of size ExE,,are concatenated into a single matrixWconca∈RE×3E. A low-rank SVD is then applied to this combined matrix to achieve compression.

$$\left[W_{q},W_{k},W_{v}\right]=W_{\text {concat}}\approx W_{r}^{d}\times Σ_{r}\times W_{r}^{u}\tag{2}$$

$$W_{qkv}^{d}=W_{r}^{d}Σ_{r}^{\beta },W_{qkv}^{u}=\left[W_{q}^{u},W_{k}^{u},W_{v}^{u}\right]=Σ_{r}^{1-\beta }W_{r}^{u}\tag{3}$$

[Wq,Wk,Wv]=Wqkvd×[Wqu..,Wku,Wvu]  (4)

whereWqkvd∈RE×r,Wqu Wku Wvu∈RrxE,and β satisfies0≤β≤1. After decomposition, the QKV components share a common down-projection matrixWqkvd,while each maintains its

<!-- 4 -->

own distinct up-projection matrix. This results in a total weight size ofαqsvd=4rE(Figure 2(c)). The computational cost for generating the query, key, and value vectors isγqv=4LE which arises from two steps: first, multiplying the input X withWqkvto generateCqkv,and then performing a second multiplication wvith the concatenated matrices[Wqu,Wku,Wvu].During inference, the intermediate productsCqkvbetween the input and the down-projection matrixWqkvare buffered to compute the KV vectors, yielding a total buffer size ofηqsvd=rL(Figure 2 (f)), and the KV vectors can be easily recomputed as:

$$K=C_{qkv}W_{k}^{u},V=C_{qkv}W_{v}^{u}\tag{5}$$

In comparison, our method achieves reduced weight size and computational cost when 4rE&lt;3E2and 4LrE&lt;3LE2,which holds when&lt;0..75E. This condition is easily satisfied with negligible accuracy loss, as demonstrated by the evaluation results in Section 4. Furthermore, our method consistently reduces the buffered size for the intermediate data, since rE is always smaller than 2E2given that&lt;E.During decoding, the cached intermediate representationCqkvis used to reconstruct the key and value matrices viaWkuandWvuwhich are then combined with the current query (sequence lengthl=1)to compute the attention outputs.

Previous methods [60,52] apply SVD individually to the weight matrices, as illustrated in Figure 2(b), resulting in a total ofαi=6rEparameters. During inference, the intermediate productsCkand Cv,which is computed from the input X and the down-projection matricesWkdandWvdmust be buffered, leading to a total buffer size ofηi=2L(Figure 2(e)). The computational cost for generating the query, key, and value vectors isγin=6LrE,and the buffer size forCkandCvis consistently larger than that required to store the unifiedCqkvin our method. Finally, our method always achieves a lower weight size, computational cost and intermediate storage.

### 3.2 Cross-layer Rank Allocation for Low-rank SVD

Performing SVD on the joint QKV weights can lead to hardware efficiency gains in both computation and storage, provided that the rank r ofWqvis sufficiently reduced without compromising the final accuracy performance. A key challenge, therefore, is determining how to truncate the singular values across all self-attention blocks in the VLM. While prior work has used Fisher information [37] to assess the importance of individual weight matrix or a group of singular values [6, 16], QSVD proposes a more effective method that evaluates the importance of each singular value in a way that minimizes degradation in model accuracy.

Given the SVD of a weight matrixW=UΣVT,it can also be expressed as a summation:W= ∑i=1uiσiviT,whereσiis the i -th singular value,and uiviare the corresponding left and right singular vectors. Truncating a singular value by settingσi=0effectively removes its associated single-rank component from the matrix, resulting in a modified representationWσi′of W,we have:

$$\Delta W_{σ_{i}}=W-W_{σ_{i}}^{\prime }=u_{i}σ_{i}v_{i}^{T}\tag{6}$$

The truncation ofσiwill affect the final output of the VLM and lead to an increase in the training lossLt. The corresponding change in training loss can be estimated through first-order expansion:

$$L_{t}\left(W_{σ_{i}}^{\prime }\right)=L_{t}\left(W-\Delta W_{σ_{i}}\right)\approx L_{t}(W)-\sum _{j,k}\Delta W_{σ_{i}}[j,k]·\frac {\partial L_{t}}{\partial W[j,k]}\tag{7}$$

whereLt(Wσi′)denotes the training loss after the weight matrix is modified toWσi′,Wσi[j,k] represents the (j,k) -th element of the matrixWσiLetGWrepresent the gradient of theloss with respect to the original weight matrix W, the changes on the training loss can be expressed as follows:

$$\Delta L_{σ_{i}}=L_{t}(W)-L_{t}\left(W_{σ_{i}}^{\prime }\right)\approx \sum _{j,k}\Delta W_{σ_{i}}[j,k]·G_{W}[j,k]=\left\langle \Delta W_{σ_{i}},G_{W}\right\rangle _{F}\tag{8}$$

where 〈·,·〉F denotes the Frobenius inner product over matrix elements. This formulation enables estimation of each singular value's contribution(e.g.,.σi)to the change in training loss, providing a principled basis for rank selection by measuring the sensitivity of the loss function to each truncated component across all layers, which can be used to evaluate the importance for each singular value. Specifically,by evaluatingΔLσiacross multiple calibration samples and computing its squared expectation, we derive the Importance Score $\hat {I}_{σ_{i}}$ for each singular valueσi,which serves as an

<!-- 5 -->

empirical approximation of the diagonal Fisher information:

$$\hat {I}_{σ_{i}}=\mathbb {E}_{x\;\mathcal {D}}\left[\left(\Delta L_{σ_{i}}^{(n)}\right)^{2}\right]\approx \frac {1}{N}\sum _{n=1}^{N}\left(\sum _{j,k}\Delta W_{σ_{i}}[j,k]·G_{W}^{(n)}[j,k]\right)\tag{9}$$

where D denotes the calibration dataset, n indexes individual samples, and N is the total number of samples in D. However, computing the importance score as defined in Equation 9 poses a significant memory burden, primarily due tothe need to construct and storeΔWσi[jk]for all singular values. Since eachΔWσiis a fullExEmatrix and there are E such singular values from the joint SVD, the total memory cost scales asO(E3)per layer, making naive computation impractical for large models. To address this, the importance score $\hat {I}_{σ_{i}}$ can alternatively be computed as follows:

$$\hat {I}_{σ_{i}}=\frac {1}{N}\sum _{n=1}^{N}σ_{i}^{2}\left[U^{T}G_{W}^{(n)}V\right]_{(i,i)}^{2}\tag{10}$$

where U and V are the left and right singular vectors from the SVD,σiis the i-th singular value. The notation (i,i) refers to the i-th diagonal element of the transformed gradient matrixUTGW()V.The proof is given in Appendix A.1. This form eliminates the need to compute and storeΔWσifor each singular value,requiring onlyO(E2)memory instead ofO(E3)per layer.

The overall SVD procedure is as follows. Starting with the original model, we first concatenate the QKV weight matrices and apply joint SVD, following the method outlined in Section 3.1. We adopt the activation-aware SVD technique from ASVD [60] to extract the singular values across all layers. For each singular value, we then compute its corresponding importance score based on the calibration dataset, as defined in Equation 10. After computing the importance scores, we perform cross-layer ranking by globally sorting all singular values based on their scores. We retain only the top k singular values with the highest importance scores, where k termed rank budget. All remaining singular values are truncated. This global ranking strategy ensures that the most critical components are preserved regardless of the layer they originate from, allowing for an optimal allocation of rank capacity throughout the VLM. For QSVD, we apply the rank selection mechanism to the self-attention layers throughout the entire VLM.

### 3.3 Post-Training Quantization Scheme for Low-Rank VLMs

<!-- L Channel Token length -->
![](./images/05f181710f3dc4f0.jpg)

<!-- 2.0 1.0 % 10 400 Channel Token length -->
![](./images/406463cdadb1319f.jpg)

Building on the efficient low-rank SVD ap-proach described in Section 3.1 and Sec-tion 3.2, this section presents an efficient quantization scheme applied to the result-ing low-rank VLMs for further hardware efficiency enhancement. To analyze the outlier distribution in the VLM, we profile the input activation distribution of LLaVA- (a) (b) (c) v1.5 13B [33]. Specifically, we examine the input activations X across the self- Figure 3: Input activation distribution within VLM. attention modules and feed-forward mod- Only partial channel are shown. ules within the language model of the

<!-- 600 500 - - 0 Channel 8 50 Token length -->
![](./images/dd2bcb6ec8e92c15.jpg)

VLM, as illustrated in Figure 3 (a) and (b), respectively. Our analysis reveals prominent chan-nelwise outliers in X across all three components, which poses the great challenge when quantizing these VLM activations for low-precision operations.

To address this issue, we adopt the rotational method introduced in [2, 54] and outlined in Section 2.3to smooth channelwise outliers. However, since the self-attention architecture of the VLM has been modified by the application of SVD, we develop an efficient quantization approach that accounts for this change. Let X denote the input to the weight matrices, and the output be expressed as Y=XWqkvWqkvu=CqkvWqkvu whereWqkv=WΣβand WVqkvu=Σ1-βWu,as notated in Equation 3. Thedistribution ofCqkv,which is buffered for KV ector recomputation, is shown in Figure 3 (c). We observe thatCqkvexhibits channelwise outliers, rendering it unsuitable for low-precision quantization. To eliminate the channelwise outliers in both X and the compressed representationCqkvwe introduce orthogonal matricesH1andH2. The self-attention computation together with its quantized counterpart are then reformulated as follows:

$$Y=\left(XH_{1}^{\top }\right)\left(H_{1}W_{qkv}^{d}H_{2}^{\top }\right)\left(H_{2}W_{qkv}^{u}\right)\quad Y^{\prime }=Q\left(C_{qkv}\right)Q\left(H_{2}W_{qkv}^{u}\right)\tag{11}$$

<!-- 6 -->

Table 1: Accuracy evaluation of different methods. For ASVD and SVDLLM, theirR1,R2are shared. Detailed results can be found in Appendix A.2.

<table border="1" ><tr>
<td rowspan="2"></td>
<td rowspan="2">Method</td>
<td colspan="8">ScienceQA-IMG↑</td>
<td colspan="4">VizWiz↑</td>
</tr><tr>
<td>Acc.</td>
<td>Hw cost</td>
<td>Acc.</td>
<td>Hw cost</td>
<td>Acc.</td>
<td>Hw cost</td>
<td>Acc.</td>
<td>Hw cost<br>心</td>
<td>Acc.</td>
<td>Hw cost</td>
<td>Acc.</td>
<td>Hw cost</td>
</tr><tr>
<td rowspan="4">SmalVILM 3</td>
<td>ASVD</td>
<td>53.84%</td>
<td>R1:100%R2:50.0%</td>
<td>7.88%</td>
<td>巴⋅:90.0%R2:42.5%</td>
<td>0.69%</td>
<td>R1:80.0%R2</td>
<td>0.10%</td>
<td>70.0%<br>R2:27.5%</td>
<td>6.68%</td>
<td>R1:100%m</td>
<td>0.00%</td>
<td>R1:80.0%M</td>
</tr><tr>
<td>SVDLLM</td>
<td>65.89%</td>
<td></td>
<td>34.61%</td>
<td>m</td>
<td>9.07%</td>
<td>35.0%</td>
<td>3.02%</td>
<td></td>
<td>14.86%</td>
<td></td>
<td>0.13%</td>
<td>:35.0%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>83.78%</td>
<td>1:100%<br>R2:37.5%</td>
<td>81.70%</td>
<td>:90.0%<br>R2:33.75%</td>
<td>79.57%</td>
<td>R1:80.0%R2:30.0%</td>
<td>77.64%</td>
<td>R1:70.0%B2:26.25%</td>
<td>40.67%</td>
<td>R2:37.5%</td>
<td>40.67%</td>
<td>:80.0%<br>R2:30.0%</td>
</tr><tr>
<td>FP16</td>
<td colspan="8">Accuracy:84.53%</td>
<td colspan="4">Accuracy:37.07%<br>心</td>
</tr><tr>
<td rowspan="4">I<br>N<br>-<br>A<br>A<br>T</td>
<td>ASVD</td>
<td>50.72%</td>
<td>R1:63.3%R2:22.5%</td>
<td rowspan="2">47.15%<br>66.14%</td>
<td rowspan="2">R1:60.0%R2:20.0%</td>
<td rowspan="2">40.26%<br>64.90%</td>
<td rowspan="2">R1:56.7%R2:17.5%R</td>
<td>25.73%</td>
<td>R1:53.3%R2:15.0%</td>
<td>47.78%</td>
<td>:63.3%</td>
<td>39.41%</td>
<td></td>
</tr><tr>
<td>SVDLLM</td>
<td>65.94%</td>
<td>a</td>
<td>62.87%</td>
<td>m</td>
<td>48.01%</td>
<td>R2:22.5%w</td>
<td>47.74%</td>
<td>R1<br>17.5%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>69.91%</td>
<td>:60.0<br>R2<br>:22.5%</td>
<td>68.22%</td>
<td>R1:53.3%<br>R2:20.0%</td>
<td>67.03%</td>
<td>:46.7%<br>R217.5%</td>
<td>65.15%</td>
<td>R2:15.0%</td>
<td>54.38%</td>
<td>:60.0%<br>B :22.5%</td>
<td>51.42%</td>
<td>R2:17.5%</td>
</tr><tr>
<td>FP16</td>
<td colspan="8">Accuracy:69.51%</td>
<td colspan="4">Accuracy:54.46%</td>
</tr><tr>
<td rowspan="4">S<br>I<br>A<br>8<br>w<br>A<br>e<br>T</td>
<td>ASVD</td>
<td>64.70%</td>
<td>R1:63.3%R2:22.5%</td>
<td rowspan="2">56.92%<br>71.44%</td>
<td rowspan="2">R1:60.0%R220.0%in</td>
<td rowspan="2">46.50%<br>71.29%<br>%</td>
<td rowspan="2">R1:56.7%R2:17.5%h</td>
<td rowspan="2">42.79%<br>70.50%</td>
<td rowspan="2">R1:53.3%R2:15.0%R1</td>
<td rowspan="2">44.48%<br>51.03%</td>
<td rowspan="2">R163.3%R2:22.5%R1</td>
<td rowspan="2">40.01%<br>49.37%</td>
<td rowspan="2">R:56.7%W <br>:17.5%</td>
</tr><tr>
<td>SVDLLM</td>
<td>71.44%</td>
<td>a</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>71.79%</td>
<td>:600<br>R2:22.5%</td>
<td>71.74%</td>
<td>:53.3%<br>R2:20.0%</td>
<td>71.74%</td>
<td>R217.5%</td>
<td>70.80%</td>
<td>R2:15.0%</td>
<td>56.15%</td>
<td>R2:22.5%</td>
<td>55.79%</td>
<td>R1:46.7%R2:17.5%</td>
</tr><tr>
<td>FP16</td>
<td colspan="8">Accuracy:71.78%</td>
<td colspan="4">Accuracy:53.63%</td>
</tr></table>

whereCqkvcan be approximately computed using the quantized input and weight as:

$$C_{qkv}\approx Q\left(XH_{1}^{\top }\right)Q\left(H_{1}W_{qkv}^{d}H_{2}^{\top }\right)\tag{12}$$

The computations presented in Equation 11 and Equation 12 support low-precision execution while reducing the size of both the weight parameters and the intermediate resultsCqkv,thereby lowering the overall memory footprint and reduce the processing latency.

Although the introduction ofH1andH2helps mitigate outliers in X andCqkv,respectively,we observe that these transformations do not fully eliminate the severe outliers, particularly those present inCqkvTo analyze this issue, we examine the distribution ofWVqkvdwhich directly influences the distribution ofCqkvWe observe thatWqkvd=WrdΣrβis strongly affected by the parameter β. Since Σris a diagonal matrix whose entries are singular values that can vary significantly in magnitude, raising them to the power β can amplify the disparity. This, in turn, exacerbates the presence of outliers inCqkv, as shown by the following derivation:

$$C_{qkv}=XW_{qkv}^{d}=XW_{r}^{d}Σ_{r}^{\beta }=XW_{r}^{d}\text {diag}\left(σ_{1}^{\beta },σ_{2}^{\beta },\cdots ,σ_{r}^{\beta }\right)=\left[σ_{1}^{\beta }\left(XW_{r}^{d}\right)_{1},\cdots ,σ_{r}^{\beta }\left(XW_{r}^{d}\right)_{r}\right]\tag{13}$$

where(XWrd)i denotes the i-th column ofXWrdwhich can significantly influence the channelwise outlier distribution inCqkv. To address this, we propose learning an optimal value for β by optimizing it over the calibration datasetD, namely:

$$\min _{\beta }\sum _{d\in D}\left\|Y_{d}-Y_{d}^{\prime }\right\|^{2}\tag{14}$$

whereYdandYd′denote the self-attention block outputs with and without quantization for the d-th sample in the calibration datasetD, respectively. The parameter β is optimized individually for each layer within the VLM. Finally, we apply the quantization operations to both the visual encoder and all layers of the language model, resulting in an end-to-end efficient VLM computation.

## 4 Evaluation Results

We evaluate QSVD on five VLMs: LLaVA-v1.5 7B [33],LLaVA-v1.5 13B,LLaVA-Next 7B,LLaVA-Next 13B, and SmolVLM-Instruct [39]. To determine the optimal rank allocation and β parameters, we use 256 samples from the ScienceQA training dataset [36],following the procedures outlined in Section 3.2 and Section 3.3. For evaluation, we adopt three widely used benchmark datasets, ScienceQA [36], VizWiz [15], and SEED-Bench-IMG [21], in line with prior work suich as LLaVA. We compare QSVD against baseline methods by implementing them on the aforementioned VLM models, the baselines include the SVD approaches (ASVD [60], SVD-LLM [52]) and quantization approach (QuaRot [2], DuQuant [29], QVLM [46]). Specifically, for ASVD and SVD-LLM,we follow their official implementations by applying SVD separately to the Key and Value matrices,while avoiding decomposition of the Query matrices to prevent performance degradation. Additionally, SVD is not applied to other linear layers within the VLM. All methods are evaluated using the same calibration samples and random seeds to ensure fairness, and we report their best performance. For QuaRot and DuQuant, we apply them to the various VLMs by strictly following the detailed procedures provided in their respective code repositories.

<!-- 7 -->

**Table 2: Quantization evaluation across different models and datasets.**R1is omitted since they are **similar for different methods. Detailed results can be found in Appendix A.2.**

<table border="1" ><tr>
<td rowspan="2">Model</td>
<td rowspan="2">Bit</td>
<td colspan="4">Duquant [29]</td>
<td colspan="4">QVLM[46]</td>
<td colspan="4">QASVD</td>
<td colspan="4">Ours</td>
</tr><tr>
<td>R2</td>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
<td>R2</td>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
<td>R2</td>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
<td>R2</td>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
</tr><tr>
<td rowspan="4">5<br>4<br>1<br>音I 1</td>
<td>FP16</td>
<td>100%</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
<td>100%</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
<td>100%</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
<td>100%</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
</tr><tr>
<td>W8A8</td>
<td>50%</td>
<td>66.53%</td>
<td>49.86%</td>
<td>58.62%</td>
<td>50%</td>
<td>64.65%</td>
<td>50.64%</td>
<td>51.82%</td>
<td>50%</td>
<td>52.95%</td>
<td>48.31%</td>
<td>53.92%</td>
<td>18.75%</td>
<td>67.57%</td>
<td>54.06%</td>
<td>60.20%</td>
</tr><tr>
<td>W8A4</td>
<td>25%</td>
<td>57.36%</td>
<td>50.07%</td>
<td>54.11%</td>
<td>25%</td>
<td>55.24%</td>
<td>48.33%</td>
<td>50.13%</td>
<td>25%</td>
<td>41.92%</td>
<td>47.85%</td>
<td>41.26%</td>
<td>9.38%</td>
<td>65.61%</td>
<td>52.18%</td>
<td>58.49%</td>
</tr><tr>
<td>W4A4</td>
<td>25%</td>
<td>52.56%</td>
<td>48.77%</td>
<td>49.50%</td>
<td>25%</td>
<td>51.12%</td>
<td>47.38%</td>
<td>34.00%</td>
<td>25%</td>
<td>12.61%</td>
<td>1.23%</td>
<td>10.48%</td>
<td>9.38%</td>
<td>55.16%</td>
<td>52.05%</td>
<td>52.69%</td>
</tr><tr>
<td rowspan="4">C<br>I<br>A<br>-<br>H<br>A<br>T</td>
<td>FP16</td>
<td>100%</td>
<td>71.78%</td>
<td>53.63%</td>
<td>62.53%</td>
<td>100%</td>
<td>71.78%</td>
<td>53.63%</td>
<td>62.53%</td>
<td>100%</td>
<td>71.78%</td>
<td>53.63%</td>
<td>62.53%</td>
<td>100%</td>
<td>71.78%</td>
<td>53.63%</td>
<td>62.53%</td>
</tr><tr>
<td>W8A8</td>
<td>50%</td>
<td>69.66%</td>
<td>50.73%</td>
<td>62.70%</td>
<td>50%</td>
<td>70.65%</td>
<td>50.32%</td>
<td>62.36%</td>
<td>50%</td>
<td>70.25%</td>
<td>54.93%</td>
<td>61.84%</td>
<td>18.75%</td>
<td>72.12%</td>
<td>55.42%</td>
<td>62.91%</td>
</tr><tr>
<td>W8A4</td>
<td>25%</td>
<td>67.22%</td>
<td>53.07%</td>
<td>61.43%</td>
<td>25%</td>
<td>66.46%</td>
<td>49.03%</td>
<td>59.22%</td>
<td>25%</td>
<td>65.34%</td>
<td>52.61%</td>
<td>59.30%</td>
<td>9.38%</td>
<td>70.12%</td>
<td>53.20%</td>
<td>62.95%</td>
</tr><tr>
<td>W4A4</td>
<td>25%</td>
<td>65.80%</td>
<td>49.37%</td>
<td>59.28%</td>
<td>25%</td>
<td>64.86%</td>
<td>48.57%</td>
<td>41.07%</td>
<td>25%</td>
<td>20.35%</td>
<td>37.5%</td>
<td>20.96%</td>
<td>9.38%</td>
<td>65.82%</td>
<td>56.82%</td>
<td>61.79%</td>
</tr><tr>
<td rowspan="4">I<br>o<br>N<br>-<br>A<br>A<br>E<br>T</td>
<td>FP16</td>
<td>100%</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
<td>100%</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
<td>100%</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
<td>100%</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
</tr><tr>
<td>W8A8</td>
<td>50%</td>
<td>66.34%</td>
<td>52.05%</td>
<td>67.91%</td>
<td>50%</td>
<td>64.70%</td>
<td>47.55%</td>
<td>66.82%</td>
<td>50%</td>
<td>64.94%</td>
<td>47.3%</td>
<td>66.87%</td>
<td>18.75%</td>
<td>69.09%</td>
<td>53.42%</td>
<td>68.92%</td>
</tr><tr>
<td>W8A4</td>
<td>25%</td>
<td>66.34%</td>
<td>50.26%</td>
<td>63.64%</td>
<td>25%</td>
<td>60.60%</td>
<td>48.55%</td>
<td>50.38%</td>
<td>25%</td>
<td>43.37%</td>
<td>48.65%</td>
<td>49.63%</td>
<td>9.38%</td>
<td>66.10%</td>
<td>53.72%</td>
<td>65.63%</td>
</tr><tr>
<td>W4A4</td>
<td>25%</td>
<td>58.37%</td>
<td>52.00%</td>
<td>62.95%</td>
<td>25%</td>
<td>55.30%</td>
<td>48.58%</td>
<td>45.24%</td>
<td>25%</td>
<td>19.17%</td>
<td>3.30%</td>
<td>13.68%</td>
<td>9.38%</td>
<td>59.67%</td>
<td>52.00%</td>
<td>62.08%</td>
</tr><tr>
<td rowspan="4">I<br>K<br>N<br>-<br>A<br>A<br>E<br>T</td>
<td>FP16</td>
<td>100%</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
<td>100%</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
<td>100%</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
<td>100%</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
</tr><tr>
<td>W8A8</td>
<td>50%</td>
<td>61.13%</td>
<td>54.38%</td>
<td>70.07%</td>
<td>50%</td>
<td>69.86%</td>
<td>49.89%</td>
<td>69.28%</td>
<td>50%</td>
<td>71.52%</td>
<td>55.13%</td>
<td>67.87%</td>
<td>18.75%</td>
<td>72.38%</td>
<td>58.33%</td>
<td>71.23%</td>
</tr><tr>
<td>W8A4</td>
<td>25%</td>
<td>70.20%</td>
<td>52.43%</td>
<td>66.15%</td>
<td>25%</td>
<td>65.28%</td>
<td>48.98%</td>
<td>65.39%</td>
<td>25%</td>
<td>64.85%</td>
<td>53.13%</td>
<td>66.54%</td>
<td>9.38%</td>
<td>70.43%</td>
<td>58.52%</td>
<td>69.21%</td>
</tr><tr>
<td>W4A4</td>
<td>25%</td>
<td>58.16%</td>
<td>53.26%</td>
<td>63.15%</td>
<td>25%</td>
<td>57.33%</td>
<td>52.23%</td>
<td>60.55%</td>
<td>25%</td>
<td>12.85%</td>
<td>4.44%</td>
<td>14.64%</td>
<td>9.38%</td>
<td>63.61%</td>
<td>54.27%</td>
<td>65.34%</td>
</tr><tr>
<td rowspan="3">E 1<br>o<br> <br>I<br>u<br>s<br>I<br>A</td>
<td>FP16</td>
<td>100%</td>
<td>84.68%</td>
<td>37.07%</td>
<td>68.18%</td>
<td>100%</td>
<td>84.68%</td>
<td>37.07%</td>
<td>68.18%</td>
<td>100%</td>
<td>84.68%</td>
<td>37.07%</td>
<td>68.18%</td>
<td>100%</td>
<td>84.68%</td>
<td>37.07%</td>
<td>68.18%</td>
</tr><tr>
<td>W8A8</td>
<td>50%</td>
<td>57.80%</td>
<td>35.52%</td>
<td>62.65%</td>
<td>50%</td>
<td>55.30%</td>
<td>33.14%</td>
<td>55.73%</td>
<td>50%</td>
<td>11.94%</td>
<td>0.00%</td>
<td>9.13%</td>
<td>18.75%</td>
<td>76.20%</td>
<td>37.82%</td>
<td>66.49%</td>
</tr><tr>
<td>W8A4</td>
<td>25%</td>
<td>55.92%</td>
<td>34.09%</td>
<td>45.82%</td>
<td>25%</td>
<td>53.52%</td>
<td>30.43%</td>
<td>42.24%</td>
<td>25%</td>
<td>3.92%</td>
<td>0.00%</td>
<td>1.23%</td>
<td>9.38%</td>
<td>62.35%</td>
<td>36.91%</td>
<td>55.35%</td>
</tr></table>

Given that QSVD performs joint SVD with quantization, we first evaluate its SVD-only performance in by comparing it with ASVD and SVD-LLM under equivalent hardware costs, including inter-mediate data storage for KV recomputation, weight size, and VLM computational cost in FLOPs. Specifically, we express it in terms of the ratio between ours and the FP16 without compression.We denote the SVD-only approach depicted in Section 4.1 as **QSVD-noQ.** We then apply the quantization techniques introduced in Section 3.3 to the SVD-compressed VLM and compare the results with advanced quantization methods such as DuQuant [29] and QVLM [46], as well as quantized version of ASVD (QASVD). For QASVD, we apply QuaRot [2] to the SVD-truncated VLMs obtained from ASVD. The corresponding results are presented in Section 4.2.

We evaluate all performance results on NVIDIA RTX A6000 GPUs using VLMEvalKit [10],and we report results under three weight-activation quantization configurations: W8A8 (8-bit weights and 8-bit activations), W8A4, and W4A4. For activation quantization, we apply per-token symmetric quantization. For weight quantization, we use round-to-nearest (RTN) with per-channel symmetric quantization and a learnable clipping ratio, determined via linear search to minimize squared error, following [2]. We present ablation studies in Section 4.3 and evaluate the latency improvements of QSVD on GPU in Section 4.4. Additional results are included in Appendix A.2.

### 4.1 Accuracy Evaluation on QSVD-noQ

We begin by evaluating the QSVD-noQ performance in FP16 under four different rank budgets k. To ensure a fair comparison, we adjust the rank configurations of all methods such that our approach consistently maintains the lowest hardware cost in terms of intermediate data storage (n),weight size (α), and computational overhead (γ), as mentionedI in Section 3.1. Importantly, as noted in Section 3.1, the relative ratios among the weight sizesαfp,aqsvd,andαiare identical to the ratios among the computationalcostsγfp ,γqsvd,andγid. This equivalence allows us to report a single normalized metric to represent both weight parameter reduction and computational efficiency. Therefore wwe haveR1andR2,defined as:

$R_{1}=\frac {α_{i}}{α_{fp}}=\frac {γ_{i}}{γ_{fp}}$ $R_{2}=\frac {η_{i}}{η_{fp}}$  (15)

where i can either be "qsvd" or "ind". The evaluation results are presented in Table 1. Our method outperforms ASVD and SVD-LLM in accuracy while incurring minimal or comparable hardware cost. On LLaVA-v1.5 13B, QSVD-noQ results in less than a 1% drop in ScienceQA-IMG accuracy compared to the original FP16 baseline, and notably even surpasses the FP16 performnance on VizWiz. For instance,withR1=4.7%and R2=17.5%,QSVD-noQachieves an accuracy of 55.79%, exceeding the FP16 counterpart by more than 2%. This may be due to the low-rank approximation effectively mitigating hallucinations [32] in the VLM; however, further investigation is needed to confirm this hypothesis. Moreover, our approach consistently achieves higher accuracy than ASVD and SVD-LLM under reduced parameter and cache ratios(R1. and.R2),with the performance gap

<!-- 8 -->

widening as these ratios decrease. For instance, in the SmolVLM setting, our method attains over 70% accuracy on ScienceQA-IMG, while both ASVD and SVD-LLM fail to operate effectively.

### 4.2 Accuracy Evaluation of QSVD

The low-rank SVD components are subsequently quantized using the techniques described in Sec-tion 3.3. We compare QSVD with DuQuant [29] and QVLM [446], as well as QASVD,which integrate QuaRot's quantization approach with the low-rank SVD outputs from ASVD,respectively. Quantization is applied consistently across the entire VLM, including both feed-forward and self-attention layers in the language model and visual encoder. Evaluations are conducted on three benchmark datasets: ScienceQA,VizWiz, and SEEDBench. All methods maintain a similar R1of approximately 50%,while RV, which has a greater impact on inference latency and cache size, varies across approaches. Notably,QSVD consistently achieves a lowerR2compared to all other baselines.

As shown in Table 2, under the W8A8 setting, QSVD consistently outperforms other baselines in most scenarios. On large-scale models like LLaVA-1.5 13B, it reaches accuracy comparable to the FP16 baseline while reducing QKV weights and compute by 50%, and cutting intermediate data size to just 18.75%. Under the more aggressive W8A4 setting, QSVD surpasses all baselines and approaches FP16-level performance using as little as 9.38% of the original KV cache. Finally, Table 2shows the quantization results under the W4A4 setting. Undler this configuration, QASVD fail to operate properly (yielding zero accuracy). Despite the challenging conditions, QSVD consistently delivers the highest performance among all models while maintaining the lowest hardware cost in terms ofR1and R2

### 4.3 Ablation Study

**Effectiveness of Cross-layer Rank Allocation Scheme** To evaluate the effectiveness of our cross-layer rank al- Table 3: Accuracy performance under location strategy (Section 3.2), we compare it with two varying rank allocation strategies. baseline methods. The first, referred to as the Uniform-rank (UR) scheme, applies SVD to the joint QKV weights using the same rank across all VLM blocks. The second, denoted as the Fisher Information-Based (FIB) scheme, also applies SVD to the joint QKV weights but distributes ranks across layers based on Fisher information. This approach has been adopted in prior work for SVD-based compression in LLMs [5]. All methods operate under the

<table border="1" ><tr>
<td></td>
<td>Method</td>
<td colspan="4">ScienceQA-IMG↑</td>
</tr><tr>
<td rowspan="6">B E<br>I<br>S<br>I<br>A<br>-<br>W<br>A<br>T 1</td>
<td>FP16</td>
<td colspan="4"></td>
</tr><tr>
<td>R1</td>
<td>60.0%</td>
<td>71.<br>53.3%</td>
<td>78%<br>46.7%</td>
<td>40.0%</td>
</tr><tr>
<td>R2</td>
<td>22.5%</td>
<td>20.0%</td>
<td>17.5%</td>
<td>15.0%</td>
</tr><tr>
<td>UR</td>
<td>71.84%</td>
<td>70.40%</td>
<td>70.40%</td>
<td>67.72%</td>
</tr><tr>
<td>FIB</td>
<td>70.60%</td>
<td>70.60%</td>
<td>70.15%</td>
<td>69.96%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>71.79%</td>
<td>71.74%</td>
<td>71.74%</td>
<td>70.80%</td>
</tr></table>

same hardware budget, defined byR1andR2. As shown in Table 3, under aggressive compression, QSVD-noQ consistently outperforms both baselines and maintains accuracy close to the FP16 model.

**Impact** **of** **Learning** β As described in Equation 14, we train β to suppress outliers in the intermediate result Cqkv Table 4 presents the impact of the learnable β on VLM accuracy under W4A4 setting over LLaVA 7Bs on Science QA. We compare QSVD with baseline methods using a fixed β across the entire VLM. QSVD consistently achieves the highest accuracy, outperforming all fixed-β

Table 4: Impact of β learning.

<table border="1" ><tr>
<td>Model</td>
<td>0.0</td>
<td>QSVD</td>
<td>0.4</td>
<td>0.8</td>
</tr><tr>
<td>v1.5-7b</td>
<td>54.53%</td>
<td>55.16%</td>
<td>54.83%</td>
<td>6.09%</td>
</tr><tr>
<td>Next-7b</td>
<td>58.80%</td>
<td>59.67%</td>
<td>56.56%</td>
<td>15.12%</td>
</tr></table>

baselines, highlighting the importance of learning β foreffective low-bit quantization.

**Long Sequence Scenarios** To further evaluate the adaptability of our QSVD method under long sequence conditions, we conduct experiments on the HRBench-4K dataset [50],which consists of 4K-resolution images. We follow the same evaluation setup as mentioned above and use VLMEvalKit [10] to report the“Aver-age All" accuracy metric. Both LLaVA-v1.6 13B and LLaVA-v1.5 13B are evalu-

Table 5: Evaluation results on HRBench-4K.

<table border="1" ><tr>
<td rowspan="2"></td>
<td rowspan="2">Method</td>
<td colspan="6">HRBench-4K↑</td>
</tr><tr>
<td>Acc.</td>
<td>Hw cost</td>
<td>Acc.</td>
<td>Hw cost<br>R</td>
<td>Acc.</td>
<td>Hw cost</td>
</tr><tr>
<td rowspan="3">I<br>N<br>-<br>出I W<br>A</td>
<td>ASVD</td>
<td>44.38%</td>
<td>R1:63.3%R2:22.5%</td>
<td>44.12%</td>
<td>:60.0%<br>R2:20.0%</td>
<td>43.12%</td>
<td>R1:56.7%R2:17.5%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>44.88%</td>
<td>R1:60.0%R2:22.5%</td>
<td>44.12%</td>
<td>1:53.3%<br>R2:20.0%</td>
<td>43.88%</td>
<td>R1:46.7%R2:17.5%</td>
</tr><tr>
<td>FP16</td>
<td colspan="6">Accuracy:45.63%</td>
</tr><tr>
<td rowspan="3">S<br>I<br>A<br>B -<br>W<br>A<br>T</td>
<td>ASVD</td>
<td>39.12%</td>
<td>R1:63.3%R2:22.5%</td>
<td>38.62%</td>
<td>R1:60.0%R2:20.0%R1:53.3%</td>
<td>36.62%</td>
<td>R1:56.7%R2:17.5%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>39.88%</td>
<td>R1:60.0%<br>R2:22.5%</td>
<td>38.75%</td>
<td>R2:20.0%</td>
<td>39.00%</td>
<td>R1:46.7%R2:17.5%</td>
</tr><tr>
<td>FP16</td>
<td colspan="6">Accuracy:39.12%</td>
</tr></table>

<!-- 9 -->

Table 6: Accuracy evaluation results (↑) on HallusionBench under different compressed parameter size ratios((R1). FP16 indicates uncompressed original models.

<table border="1" ><tr>
<td rowspan="2">R1</td>
<td colspan="4">LLaVA-v1.513B</td>
<td colspan="4">LLaVA-Next 13B</td>
</tr><tr>
<td>aAcc</td>
<td>fAcc</td>
<td>qAcc</td>
<td>Overall</td>
<td>aAcc</td>
<td>fAcc</td>
<td>qAcc</td>
<td>Overall</td>
</tr><tr>
<td>90%</td>
<td>49.63%</td>
<td>21.10%</td>
<td>17.58%</td>
<td>29.44%</td>
<td>57.73%</td>
<td>26.01%</td>
<td>26.59%</td>
<td>36.78%</td>
</tr><tr>
<td>80%</td>
<td>48.90%</td>
<td>20.52%</td>
<td>16.92%</td>
<td>28.78%</td>
<td>58.25%</td>
<td>26.01%</td>
<td>26.81%</td>
<td>37.03%</td>
</tr><tr>
<td>70%</td>
<td>50.26%</td>
<td>22.83%</td>
<td>17.80%</td>
<td>30.30%</td>
<td>58.46%</td>
<td>26.88%</td>
<td>27.25%</td>
<td>37.53%</td>
</tr><tr>
<td>FP16</td>
<td>44.69%</td>
<td>19.36%</td>
<td>16.04%</td>
<td>26.70%</td>
<td>56.78%</td>
<td>26.01%</td>
<td>25.27%</td>
<td>36.02%</td>
</tr></table>

ated under our QSVD-noQ configuration and compared against ASVD and FP16 baselines. The results are summarized in Table 5.

As shown in Table 5, QSVD-noQ consistently outperforms ASVD in all evaluation settings. Moreover, the relative performance trends on HRBench-4K closely mirror those observed on ScienceQA-IMG and VizWiz, indicating that our rank allocation strategy generalizes effectively to long sequence scenarios resulting from high-resolution visual inputs.

**Impact** of **QSVD** **on** **Hallucination** We further evaluate the imnpact of QSVD on VLM hallucination using HallusionBench [14], following the same evaluation setup as mentioned above. Metrics include aAcc,fAcc,and qAcc from HallusionBench and their overall average score. As shown in Table 6, both LLaVA-v1.5 13B and LLaVA-Next 13B exhibit noticeable improvements in groundedness metrics after QSVD-noQ. For LLaVA-Next 13B, the overall score increases from 36.02 to a peak of 37.53atR1=70%. Similarly, LLaVA-v1.5 13B improves from 26.70 to 30.30 atR1=70%,marking a clear reduction in hallucination. These findings confirm that QSVD-noQ not only reduces model and cache size but also acts as an effective regularizer against hallucinations. This effect explains why,on certain benchmark datasets such as VizWiz, the QSVD-compressed models occasionally outperform their original FP16 counterparts in terms of end-task accuracy.

### 4.4 Latency Evaluation on VLM

QSVD leverages both SVD and quantization to jointly compress model weights and KV cache, making it well-suited for deployment on memory-constrained hardware. We evaluate inference latency of the layer-wise LLaVA-v1.5 7B on an NVIDIA RTX 4070 GPU with 12GB memory.The batch size is set to 1 and the token length to 4K. As shown in Figure 4, under FP16 precision, due to limited GPU memory, both the FP16 baseline and QSVD-noQ require partial offloading to CPU memory. However, QSVD-noQ with 40% and Figure 4: Normalized speedup 30% (denoted as noQ-40%and noQ-30%) rank retention benefits of QSVD-noQ and QSVD from reduced data movement enabled by effective SVD compression, W8A8 on low-end GPU. achieving up to a2.1×speedup over the baseline. Furthermore, QSVD with W8A8 quantization, under 70% and 50% rank retention,

<!-- FP16 noQ-40% noQ-30% INT8 70% 50% 15 12.7×13.1× xx 10 9.7x Cnpee0 S 5 1x 1.7x2.1x 1.7×2.1× 0 -->
![](./images/f9f730b89e0e4eaf.jpg)

completely avoids offloading and achieves a significant speedup of up to 13.1x.

## 5 Conclusion and Limitation

In this work, we proposed QSVD, a unified framework that applies joint singular value decomposition and quantization to compress VLMs efficiently. Future work will explore joint optimization across all model blocks. Additionally, improving VLM efficiency may also make powerful models more accessible, which raises concerns about potential misuse in areas such as surveillance, misinformation, and privacy violations. Further investigation is needed to address these risks.

# Acknowledgments

This work was supported through a research collaboration with the Reality Labs Silicon at Meta Reality Labs. We sincerely thank the Silicon AI Research team, particularly Lita Yang and Steven Li, for their valuable contributions to the technical discussions and the development of the ideas presented in this paper.

<!-- 10 -->

# NeurIPS Paper Checklist

## 1.Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer:[Yes]

Justification: Yes, our abstract and introduction accuratelyreflect the main contributions and scope of the paper. We clearly state that the work proposes a novel application of Singular Value Decomposition (SVD) to the joint QKV matrices in Vision-Language Models (VLMs), with the goal of reducing KV cache size and computational overhead. This aligns with the technical content, which introduces a dynamic SVD rank allocation strategy to balance memory and accuracy. Furthermore, the abstract discusses the extension of this approach through quantizationof both weights and activations, which is supported by experimental results in the paper showing better or comparable performance against previous methods.

### Guidelines:

·The answer NA means that the abstract and introduction do not include the claims made in the paper.

·The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

·The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

·It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2.Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer:[Yes]

Justification: We discuss the limitation in the last section of the paper.

Guidelines:

·The answer NA means that the paper has no limitatioon while the answer No means that the paper has limitations, but those are not discussed in the paper.

·The authors are encouraged to create a separate "Limitations" section in their paper.

·The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

·The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

·The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

·The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

·If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

·While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor-tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

<!-- 15 -->

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer:[Yes]

Justification: We will inlcude full derivation of our importance score in Appendix A.1.

### Guidelines:

·The answer NA means that the paper does not include theoretical results.

·All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

·All assumptions should be clearly stated or referenced in the statement of any theorems.

·The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

·Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

·Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Yes, the paper discloses sufficient information to reproduce its main results. It clearly describes the methods for joint QKV SVD, rank allocation, and quantization,with supporting equations and implementation details. The evaluation setup, including models, datasets,and calibration procedures, is well-documented.

### Guidelines:

·The answer NA means that the paper does not include experiments.

·If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

·If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

·Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

·While NeurIPS does not require releasing code, the conference does require all submis-sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c)If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

<!-- 16 -->

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer:[Yes]

Justification: We submit the codeas the supplementary materials.

Guidelines:

·The answer NA means that paper does not include experiments requiring code.

·Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.

·While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

·The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

·The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

·The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible,they should state which ones are omitted from the script and why.

·At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

·Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We include necessary experiments specifications in supplementary materials Guidelines:

·The answver NA means that the paper does not include experiments.

·The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

·The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer:[Yes]

Justification: Yes, we report the average results over 5 random seeds in all our quanization part. And for calculation of our reported accuracy we follow the opensource VLM evaluation framework.

Guidelines:

<!-- 17 -->

·The answver NA means that the paper does not include experiments.

·The authors should answer "Yes" if the results are accompanied by error bars, confi-dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

·The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

·The method for calculating the error bars should be explained (closed form formula, call to a library function,bootstrap,etc.)

·The assumptions made should be given (e.g., Normally distributed errors).

·It should be clear whether the error bar is the standard deviation or the standard error of the mean.

·It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI,if the hypothesis of Normality of errors is not verified.

·For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

·If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the com-puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer:[Yes]

Justification: All the details are provided in the supplementary materials.

Guidelines:

·The answer NA means that the paper does not include experiments.

·The paper should indicate the type of compute workers CPU or GPU,internal cluster, or cloud provider, including relevant memory and storage.

·The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

·The paper should disclose wvhether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9.Code of ethics

Question: Does the research conducted in the paper conform, in every respect,with the NeurIPS CCode of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer:[Yes]

Justification: We have read and acknowledge the NeurIPS Code of Ethics.

Guidelines:

·The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

·If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

·The authors should make sure to preserve anonymity (e.g.,if there is a special consid-eration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal immpacts of the work performed?

Answer:[Yes]

Justification: We mentioned it in the last section of the paper.

<!-- 18 -->

### Guidelines:

·The answer NA means that there is no societal impact of the work performed.

·If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

·Examples of negative societal impacts include potential malicious or unintended uses (e.g.,disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

·The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

·The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

·If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechannisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11.Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our paper does not involve the release of data or models that carry a high risk of misuse,and therefore no specific safeguards are necessary in this context.

### Guidelines:

·The answer NA means that the paper poses no such risks.

·Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

·Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

·We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faitheffort.

## 12.Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer:[Yes]

Justification: All code, data, models utilized in this paper are cited and credited.

Guidelines:

·The answer NA means that the paper does not use existing assets.

·The authors should cite the original paper that produced the code package or dataset.

·The authors should state which version of the asset is used and, if possible, include a URL.

·The name of the license (e.g., CC-BY 4.0) should be included for each asset.

<!-- 19 -->

·For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

·If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

·For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

·If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: Our paper does not release new models and assets.

Guidelines

·The answer NA means that the paper does not release new assets.

·Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations,etc.

·The paper should discuss whether and how consent was obtained from people whose asset is used.

·At submission time, remember to anonymize your assets (if applicable).You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research wwith human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable,as well as details aabout compensation (if any)?

Answer: [NA]

Justification: Our paper do not involve human subjects.

Guidelines:

·The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

·Including this information in the supplemental material is fine, but if the main contribu-tion of the paper involves human subjects, then as much detail as possible should be included in the main paper.

·According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants,whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) wvere obtained?

Answer:[NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects

Guidelines: The paperdoes not involve crowdsourcing nor research with human subjects.

·The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

<!-- 20 -->

·Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval,you should clearly state this in the paper.

·We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

·For initial submissions, do not include any information that would break anonymity(if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This work does not use large language models (LLMs) as an important, original, or non-standard component of the core methodology. LLMs were only used minimally for editing or polishing writing and did not influence the research's scientific content, methodology, or conclusions.

## Guidelines:

·The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

·Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

<!-- 21 -->

# A Technical Appendices and Supplementary Material

# Outline of Appendices

A.1: Detailed derivation for Importance Score in Sec. 3.2.

A.2: Additional experimental results.

A.3:Case study for QSVD

## A.1 Importance Score derivation

**Proof.** Recall that the singular value decomposition (SVD) ofW is given by:

$$W=UΣV^{T}=\sum _{i=1}^{r}σ_{i}u_{i}v_{i}^{T}$$

whereui and viare the i-th left and right singular vectors, andσiis the i-th singular value. When truncatingσi,the change in Wis:

$$\Delta W_{σ_{i}}=σ_{i}u_{i}v_{i}^{T}$$

Then, the inner product betweenΔWiand the gradientGW(n)is:

$$\sum _{j,k}\Delta W_{σ_{i}}(j,k)·G_{W}^{(n)}(j,k)=\left\langle \Delta W_{σ_{i}},G_{W}^{(n)}\right\rangle _{F}$$

where 〈·,·〉Fdenotes the Frobenius inner product.

SubstituteΔWσi=σiuiviT

$$\left\langle \Delta W_{σ_{i}},G_{W}^{(n)}\right\rangle _{F}=\left\langle σ_{i}u_{i}v_{i}^{T},G_{W}^{(n)}\right\rangle _{F}=σ_{i}\left\langle u_{i}v_{i}^{T},G_{W}^{(n)}\right\rangle _{F}$$

Using the property of the Frobenius inner product:

$$\langle A,B\rangle _{F}=\text {tr}\left(A^{T}B\right)$$

we have:

$$σ_{i}\left\langle u_{i}v_{i}^{T},G_{W}^{(n)}\right\rangle _{F}=σ_{i}\text {tr}\left(\left(u_{i}v_{i}^{T}\right)^{T}G_{W}^{(n)}\right)=σ_{i}\text {tr}\left(v_{i}u_{i}^{T}G_{W}^{(n)}\right)$$

By cyclic property of trace:

$$σ_{i}\text {tr}\left(v_{i}u_{i}^{T}G_{W}^{(n)}\right)=σ_{i}\text {tr}\left(u_{i}^{T}G_{W}^{(n)}v_{i}\right)$$

SinceuiTGW(n)viis a scalar:

$$σ_{i}\text {tr}\left(u_{i}^{T}G_{W}^{(n)}v_{i}\right)=σ_{i}\left(u_{i}^{T}G_{W}^{(n)}v_{i}\right)$$

Note that:

$$U^{T}G_{W}^{(n)}V\in \mathbb {R}^{r\times r}$$

and the (i,i))-th diagonal entry is:

<!-- 22 -->

$$\left[U^{T}G_{W}^{(n)}V\right]_{(i,i)}=u_{i}^{T}G_{W}^{(n)}v_{i}$$

Therefore:

$$\sum _{j,k}\Delta W_{σ_{i}}(j,k)·G_{W}^{(n)}(j,k)=σ_{i}\left[U^{T}G_{W}^{(n)}V\right]_{(i,i)}$$

$$\sum _{j,k}\Delta W_{σ_{i}}(j,k)·G_{W}(j,k)=\left\langle \Delta W_{σ_{i}},G_{W}\right\rangle _{F}=σ_{i}\left[U^{T}G_{W}V\right]_{(i,i)}$$

Finally, the importance score $\hat {I}_{σ_{i}}$ can be computed as follows:

$$\hat {I}_{σ_{i}}=\frac {1}{N}\sum _{n=1}^{N}σ_{i}^{2}\left[U^{T}G_{W}^{(n)}V\right]_{(i,i)}^{2}$$

If a whitening transformation such as ASVD or SVD-LLM is applied prior to SVD,that is, UΣVT=SV(WS)where S denotes the whitening matrix, the corresponding importance score can be reformulated as:

$$\hat {I}_{σ_{i}}=\frac {1}{N}\sum _{n=1}^{N}σ_{i}^{2}\left[U^{T}G_{W}^{(n)}S^{-T}V\right]_{(i,i)}^{2}$$

where the termS-Tconverts the gradient from the original weight space to the whitened space in which the SVD is performed.

## A.2 Full table for experiments

Here we include more detailed experiments table for QSVD method. We evaluate all performance reuslts on NVIDIA RTX A6000 GPUs, we report results under QSVD-noQ and under three weight-activation quantization configurations: W8A8 (8-bit weights and 8-bit activations),W8A4,and W4A4. For activation quantization, we apply per-token symmetric quantization. For weight quantization, we use round-to-nearest (RTN) with per-channel symmetric quantization and a learnable clipping ratio, determined via linear search to minimize squared error, following [2]. For QSVD baseline,we add QuaRot without SVD as an baseline, also for ours method, we use activation clip ratio of 0.85 for vit model and 0.9 for language model, under this setting, we have updated some QSVD accuracy results higher than main paper report.

**QSVD-noQ** **results.** Table 7 presents detailed results of QSVD-noQ on SmolVLM [39], LLaVA-v1.5 [33] series, and LLaVA-Next series, along with the corresponding preserved ratios. QSVD-noQ consistently outperforms ASVD and SVD-LLM in accuracy under reduced parameter and cache ratios (R1.and.R2),with the performance gap widening as the compression becomes more aggressive. For instance, in the SmolVLM setting, our method maintains over 70% accuracy on ScienceQA-IMG [36], while both ASVD [60] and SVD-LLM [51] fail to function effectively.

**QSVD** **Results.** We evaluate our proposed QSVD quantization strategy on LLaVA v1.5 and Next series: 7B, 13B,across three benchmarks: ScienceQA, VizWiz, and SEEDBench. Table 8 summarizes performance under two low-bit settings, W8A8 and W8A4.

Under the W8A8 setting, QSVD matches or exceeds prior methods such as Duquant [29], QVLM [46], and QASVD,while reducing the KV cache and intermediate data sizes by up to 50%.Compared to QSVDLLM, our approach avoids the need for manually re-optimizing decomposed matrices while still achieving superior performance.

In the more challenging W8A4 configuration, QSVD continues to deliver robust outputs,reaching levels comparable to the FP16 baseline using just 9.38% of the original KV cache. This demonstrates the scalability of our quantization design under aggressive memory constraints.

<!-- 23 -->

<table border="1" ><tr>
<td></td>
<td>Method</td>
<td colspan="4">ScienceQA-IMG↑</td>
<td colspan="4">VizWiz↑</td>
</tr><tr>
<td rowspan="5">8<br>Z<br>-<br>I<br>N<br>T<br>A<br>I<br>o<br>u<br>S</td>
<td>FP16</td>
<td colspan="4">84.53%</td>
<td colspan="4">37.07%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>100%<br>50.0%</td>
<td>90.0%<br>42.5%</td>
<td>80.0%<br>35.0%</td>
<td>70.0%<br>27.5%</td>
<td>100%<br>50.0%</td>
<td>90.0%<br>42.5%</td>
<td>80.0%<br>35.0%</td>
<td>70.0%<br>27.5%</td>
</tr><tr>
<td>ASVD<br>SVDLLM</td>
<td>53.84%<br>65.89%</td>
<td>7.88%<br>34.61%</td>
<td>0.69%<br>9.07%</td>
<td>0.10%<br>3.02%</td>
<td>6.68%<br>14.86%</td>
<td>0.00%<br>1.62%</td>
<td>0.00%<br>0.13%</td>
<td>0.00%<br>0.00%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>100%<br>37.5%</td>
<td>90.0%<br>33.75%</td>
<td>80.0%<br>30.0%</td>
<td>70.0%<br>26.25%</td>
<td>100%<br>37.5%</td>
<td>90.0%<br>33.75%</td>
<td>80.0%<br>30.0%</td>
<td>70.0%<br>26.25%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>83.78%</td>
<td>81.70%</td>
<td>79.57%</td>
<td>77.64%</td>
<td>40.67%</td>
<td>39.88%</td>
<td>40.67%</td>
<td>43.84%</td>
</tr><tr>
<td rowspan="6">8<br>L<br>S<br>I<br>A<br>-<br>h<br>A<br>e<br>T<br>T</td>
<td>FP16</td>
<td colspan="4">68.01%</td>
<td colspan="4">50.03%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
</tr><tr>
<td>ASVD<br>SVDLLM</td>
<td>22.36%<br>55.23%</td>
<td>19.09%<br>55.03%</td>
<td>15.22%<br>49.23%</td>
<td>10.81%<br>51.17%</td>
<td>47.70%<br>50.10%</td>
<td>39.02%<br>50.71%</td>
<td>10.10%<br>50.47%</td>
<td>8.87%<br>49.99%</td>
</tr><tr>
<td rowspan="2">R1<br>R2</td>
<td>60.0%</td>
<td>53.3%</td>
<td rowspan="2">46.7%<br>17.5%</td>
<td rowspan="2">40.0%<br>15.0%</td>
<td rowspan="2">60.0%<br>22.5%</td>
<td rowspan="2">53.3%<br>20.0%</td>
<td rowspan="2">46.7%<br>17.5%</td>
<td rowspan="2">40.0%<br>15.0%</td>
</tr><tr>
<td>22.5%</td>
<td>20.0%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>66.12%</td>
<td>65.64%</td>
<td>64.06%</td>
<td>61.68%</td>
<td>53.84%</td>
<td>54.19%</td>
<td>53.88%</td>
<td>52.40%</td>
</tr><tr>
<td rowspan="5">8<br>S<br>I<br>-<br>S<br>I<br>A<br>-<br>A<br>A<br>e<br>T<br>1</td>
<td>FP16</td>
<td colspan="4">71.78%</td>
<td colspan="4">53.63%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
</tr><tr>
<td>ASVD<br>SVDLLM</td>
<td>64.70%<br>71.44%</td>
<td>56.92%<br>71.44%</td>
<td>46.50%<br>71.29%</td>
<td>42.79%<br>70.50%</td>
<td>44.48%<br>51.03%</td>
<td>40.63%<br>51.15%</td>
<td>40.01%<br>49.37%</td>
<td>37.87%<br>46.49%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>60.0%<br>22.5%</td>
<td>53.3%<br>20.0%</td>
<td>46.7%<br>17.5%</td>
<td>40.0%<br>15.0%</td>
<td>60.0%<br>22.5%</td>
<td>53.3%<br>20.0%</td>
<td>46.7%<br>17.5%</td>
<td>40.0%<br>15.0%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>71.79%</td>
<td>71.74%</td>
<td>71.74%</td>
<td>70.80%</td>
<td>56.15%</td>
<td>56.05%</td>
<td>55.79%</td>
<td>54.04%</td>
</tr><tr>
<td rowspan="6">8<br>L<br> <br>I<br>x<br>e<br>N<br>-<br>A<br>A<br>e<br>T<br>T</td>
<td>FP16</td>
<td colspan="4">69.51%</td>
<td colspan="4">54.46%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
</tr><tr>
<td>ASVD<br>SVDLLM</td>
<td>50.72%<br>65.94%</td>
<td>47.15%<br>66.14%</td>
<td>40.26%<br>64.90%</td>
<td>25.73%<br>62.87%</td>
<td>47.78%<br>48.01%</td>
<td>47.3%<br>48.41%</td>
<td>39.41%<br>47.74%</td>
<td>6.69%<br>47.73%</td>
</tr><tr>
<td rowspan="2">R1<br>R2</td>
<td>60.0%</td>
<td rowspan="2">53.3%<br>20.0%</td>
<td rowspan="2">46.7%<br>17.5%</td>
<td rowspan="2">40.0%<br>15.0%</td>
<td rowspan="2">60.0%<br>22.5%</td>
<td rowspan="2">53.3%<br>20.0%</td>
<td rowspan="2">46.7%<br>17.5%</td>
<td rowspan="2">40.0%<br>15.0%</td>
</tr><tr>
<td>22.5%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>69.91%</td>
<td>68.22%</td>
<td>67.03%</td>
<td>65.15%</td>
<td>54.38%</td>
<td>52.31%</td>
<td>51.42%</td>
<td>49.86%</td>
</tr><tr>
<td rowspan="5">8<br>S<br>I<br>-<br>I<br>X<br>a<br>N<br>-<br>W<br>A<br>E<br>T<br>T</td>
<td>FP16</td>
<td colspan="4">73.23%</td>
<td colspan="4">57.72%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
<td>63.3%<br>22.5%</td>
<td>60.0%<br>20.0%</td>
<td>56.7%<br>17.5%</td>
<td>53.3%<br>15.0%</td>
</tr><tr>
<td>ASVD<br>SVDLLM</td>
<td>69.71%<br>70.30%</td>
<td>68.86%<br>69.71%</td>
<td>67.43%<br>69.56%</td>
<td>64.01%<br>68.52%</td>
<td>55.42%<br>53.08%</td>
<td>54.97%<br>52.54%</td>
<td>54.50%<br>52.52%</td>
<td>52.95%<br>51.77%</td>
</tr><tr>
<td>R1<br>R2</td>
<td>60.0%<br>22.5%</td>
<td>53.3%<br>20.0%</td>
<td>46.7%<br>17.5%</td>
<td>40.0%<br>15.0%</td>
<td>60.0%<br>22.5%</td>
<td>53.3%<br>20.0%</td>
<td>46.7%<br>17.5%</td>
<td>40.0%<br>15.0%</td>
</tr><tr>
<td>QSVD-noQ</td>
<td>72.63%</td>
<td>72.29%</td>
<td>72.34%</td>
<td>71.64%</td>
<td>55.48%</td>
<td>55.14%</td>
<td>54.99%</td>
<td>55.77%</td>
</tr></table>

Table 7: Accuracy on ScienceQA-IMG and VizWiz datasets. TheR1,R2denotes the proportion of preserved QKV parameters and the corresponding cache ratio.

For completeness, Appendix Tables 8 and 9 report full results across all model variants and bitwidth configurations. Notably,our method consistently ranks highest or near-highest across settings, while maintaining favorable compression ratios of 50%/18.75% in W8A8 and 50%/9.38% in W8A4. These results highlight the ability of QSVD to balance compression and output quality across a diverse range of architectures and tasks.

<!-- 24 -->

<table border="1" ><tr>
<td rowspan="2"></td>
<td rowspan="2">Method</td>
<td colspan="3">W8A8</td>
<td rowspan="2">R1/R2</td>
<td colspan="3">W8A4</td>
<td rowspan="2">R1/R2</td>
</tr><tr>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
<td>SciQA↑</td>
<td>VizWiz↑</td>
<td>SEED↑</td>
</tr><tr>
<td rowspan="7">8<br>L<br>-<br>S<br>I<br>-<br>w<br>A<br>E<br>T<br>1</td>
<td>FP16</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
<td>100%/100%</td>
<td>68.01%</td>
<td>50.03%</td>
<td>60.18%</td>
<td>100%/100%</td>
</tr><tr>
<td>QuaRot</td>
<td>67.90%</td>
<td>49.95%</td>
<td>60.11%</td>
<td>50%/50%</td>
<td>63.19%</td>
<td>49.82%</td>
<td>58.18%</td>
<td>50%/25%</td>
</tr><tr>
<td>Duquant</td>
<td>66.53%</td>
<td>49.86%</td>
<td>58.62%</td>
<td>50.52%/50%</td>
<td>57.36%</td>
<td>50.07%</td>
<td>54.11%</td>
<td>50.52%/25%</td>
</tr><tr>
<td>QVLM</td>
<td>64.65%</td>
<td>50.64%</td>
<td>51.82%</td>
<td>50%/50%</td>
<td>55.24%</td>
<td>48.33%</td>
<td>50.13%</td>
<td>50%/25%</td>
</tr><tr>
<td>QASVD</td>
<td>52.95%</td>
<td>48.31%</td>
<td>53.92%</td>
<td>50%/50%</td>
<td>41.92%</td>
<td>47.85%</td>
<td>41.26%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVDLLM</td>
<td>66.14%</td>
<td>51.93%</td>
<td>56.47%</td>
<td>50%/50%</td>
<td>30.38%</td>
<td>45.00%</td>
<td>37.00%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVD</td>
<td>67.57%</td>
<td>54.06%</td>
<td>60.20%</td>
<td>50%/18.75%</td>
<td>65.61%</td>
<td>52.18%</td>
<td>58.49%</td>
<td>50%/9.38%</td>
</tr><tr>
<td rowspan="7">8<br>S<br>I<br>S<br>T<br>-<br>A<br>A<br>E<br>T<br>1</td>
<td>FP16</td>
<td>71.80%</td>
<td>53.63%</td>
<td>62.54%</td>
<td>100%/100%</td>
<td>71.80%</td>
<td>53.63%</td>
<td>62.54%</td>
<td>100%/100%</td>
</tr><tr>
<td>QuaRot</td>
<td>71.64%</td>
<td>53.64%</td>
<td>62.57%</td>
<td>50%/50%</td>
<td>68.02%</td>
<td>54.57%</td>
<td>58.53%</td>
<td>50%/25%</td>
</tr><tr>
<td>Duquant</td>
<td>69.66%</td>
<td>50.73%</td>
<td>62.70%</td>
<td>51.67%/50%</td>
<td>67.22%</td>
<td>53.07%</td>
<td>61.43%</td>
<td>51.67%/25%</td>
</tr><tr>
<td>QVLM</td>
<td>70.65%</td>
<td>50.32%</td>
<td>62.36%</td>
<td>50%/50%</td>
<td>66.46%</td>
<td>49.03%</td>
<td>59.22%</td>
<td>50%/25%</td>
</tr><tr>
<td>QASVD</td>
<td>70.25%</td>
<td>54.93%</td>
<td>61.84%</td>
<td>50%/50%</td>
<td>65.34%</td>
<td>52.61%</td>
<td>59.30%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVDLLM</td>
<td>70.65%</td>
<td>56.32%</td>
<td>62.35%</td>
<td>50%/50%</td>
<td>60.20%</td>
<td>50.52%</td>
<td>55.03%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVD</td>
<td>72.12%</td>
<td>55.42%</td>
<td>62.91%</td>
<td>50%/18.75%</td>
<td>70.12%</td>
<td>53.20%</td>
<td>62.95%</td>
<td>50%/9.38%</td>
</tr><tr>
<td rowspan="7">H L<br>-<br>I<br>X<br>a<br>N<br>-<br>W<br>A<br>E<br>T<br>7</td>
<td>FP16</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
<td>100%/100%</td>
<td>69.60%</td>
<td>54.46%</td>
<td>69.02%</td>
<td>100%/100%</td>
</tr><tr>
<td>QuaRot</td>
<td>69.19%</td>
<td>52.86%</td>
<td>65.60%</td>
<td>50%/50%</td>
<td>64.53%</td>
<td>51.27%</td>
<td>65.08%</td>
<td>50%/25%</td>
</tr><tr>
<td>Duquant</td>
<td>66.34%</td>
<td>52.05%</td>
<td>67.91%</td>
<td>50.52%/50%</td>
<td>66.34%</td>
<td>50.26%</td>
<td>63.64%</td>
<td>50.52%/25%</td>
</tr><tr>
<td>QVLM</td>
<td>64.70%</td>
<td>47.55%</td>
<td>66.82%</td>
<td>50%/50%</td>
<td>60.60%</td>
<td>48.55%</td>
<td>50.38%</td>
<td>50%/25%</td>
</tr><tr>
<td>QASVD</td>
<td>64.94%</td>
<td>47.30%</td>
<td>66.87%</td>
<td>50%/50%</td>
<td>43.37%</td>
<td>48.65%</td>
<td>49.63%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVDLLM</td>
<td>64.70%</td>
<td>47.55%</td>
<td>66.83%</td>
<td>50%/50%</td>
<td>33.83%</td>
<td>46.05%</td>
<td>39.08%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVD</td>
<td>69.09%</td>
<td>53.42%</td>
<td>68.92%</td>
<td>50%/18.75%</td>
<td>66.10%</td>
<td>53.72%</td>
<td>65.63%</td>
<td>50%/9.38%</td>
</tr><tr>
<td rowspan="7">8<br>S<br>I<br>-<br>I<br>x<br>a<br>N<br>-<br>W<br>A<br>3<br>1</td>
<td>FP16</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
<td>100%/100%</td>
<td>73.23%</td>
<td>57.72%</td>
<td>71.30%</td>
<td>100%/100%</td>
</tr><tr>
<td>QuaRot</td>
<td>72.04%</td>
<td>58.03%</td>
<td>67.29%</td>
<td>50%/50%</td>
<td>66.98%</td>
<td>55.56%</td>
<td>70.15%</td>
<td>50%/25%</td>
</tr><tr>
<td>Duquant</td>
<td>61.13%</td>
<td>54.38%</td>
<td>70.07%</td>
<td>51.67%/50%</td>
<td>70.20%</td>
<td>52.43%</td>
<td>66.15%</td>
<td>51.67%/25%</td>
</tr><tr>
<td>QVLM</td>
<td>69.86%</td>
<td>49.89%</td>
<td>69.28%</td>
<td>50%/50%</td>
<td>65.28%</td>
<td>48.98%</td>
<td>65.39%</td>
<td>50%/25%</td>
</tr><tr>
<td>QASVD</td>
<td>71.52%</td>
<td>55.13%</td>
<td>67.87%</td>
<td>50%/50%</td>
<td>64.85%</td>
<td>53.13%</td>
<td>66.54%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVDLLM</td>
<td>69.85%</td>
<td>49.89%</td>
<td>69.27%</td>
<td>50%/50%</td>
<td>61.25%</td>
<td>45.05%</td>
<td>65.03%</td>
<td>50%/25%</td>
</tr><tr>
<td>QSVD</td>
<td>72.38%</td>
<td>58.33%</td>
<td>71.23%</td>
<td>50%/18.75%</td>
<td>70.43%</td>
<td>58.52%</td>
<td>69.21%</td>
<td>50%/9.38%</td>
</tr></table>

Table 8: Quantization results on W8A8 and W8A4.

<table border="1" ><tr>
<td rowspan="2"></td>
<td colspan="2"></td>
<td colspan="3">LLaVA-V1.5 Series</td>
<td colspan="3">LLaVA-Next Series</td>
<td rowspan="2">R1/R2</td>
</tr><tr>
<td>Bit</td>
<td>Method</td>
<td>ScienceQA↑</td>
<td>SEED↑</td>
<td>VizWiz↑</td>
<td>ScienceQA↑</td>
<td>SEED↑</td>
<td>VizWiz↑</td>
</tr><tr>
<td rowspan="7"></td>
<td></td>
<td>FP16</td>
<td>68.01%</td>
<td>60.18%</td>
<td>50.03%</td>
<td>69.60%</td>
<td>69.02%</td>
<td>54.46%</td>
<td>100%/100%</td>
</tr><tr>
<td>W4A4</td>
<td>QuaRot</td>
<td>49.08%</td>
<td>50.54%</td>
<td>49.96%</td>
<td>55.57%</td>
<td>59.81%</td>
<td>55.25%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>Duquant</td>
<td>52.56%</td>
<td>49.51%</td>
<td>48.77%</td>
<td>58.36%</td>
<td>62.95%</td>
<td>52.00%</td>
<td>27.08%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QVLM</td>
<td>51.12%</td>
<td>34.00%</td>
<td>47.38%</td>
<td>55.30%</td>
<td>45.24%</td>
<td>48.58%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QASVD</td>
<td>12.61%</td>
<td>10.48%</td>
<td>1.23%</td>
<td>19.17%</td>
<td>13.68%</td>
<td>3.30%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QSVDLLM</td>
<td>6.18%</td>
<td>5.53%</td>
<td>0.00%</td>
<td>10.13%</td>
<td>8.64%</td>
<td>2.55%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QSVD</td>
<td>55.16%</td>
<td>52.70%</td>
<td>52.05%</td>
<td>59.67%</td>
<td>62.97%</td>
<td>52.00%</td>
<td>25%/9.38%</td>
</tr><tr>
<td rowspan="7">I</td>
<td></td>
<td>FP16</td>
<td>71.80%</td>
<td>62.54%</td>
<td>53.63%</td>
<td>73.23%</td>
<td>71.30%</td>
<td>57.72%</td>
<td>100%/100%</td>
</tr><tr>
<td>W4A4</td>
<td>QuaRot</td>
<td>62.74%</td>
<td>60.14%</td>
<td>55.62%</td>
<td>57.47%</td>
<td>62.95%</td>
<td>50.13%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>Duquant</td>
<td>65.80%</td>
<td>59.28%</td>
<td>49.37%</td>
<td>58.16%</td>
<td>63.15%</td>
<td>53.26%</td>
<td>26.67%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QVLM</td>
<td>64.86%</td>
<td>41.07%</td>
<td>48.57%</td>
<td>57.33%</td>
<td>60.55%</td>
<td>52.23%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QASVD</td>
<td>20.35%</td>
<td>37.5%</td>
<td>20.96%</td>
<td>12.85%</td>
<td>14.64%</td>
<td>4.44%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QSVDLLM</td>
<td>10.53%</td>
<td>7.65%</td>
<td>1.01%</td>
<td>15.57%</td>
<td>10.89%</td>
<td>2.22%</td>
<td>25%/25%</td>
</tr><tr>
<td>W4A4</td>
<td>QSVD</td>
<td>65.82%</td>
<td>61.79%</td>
<td>56.82%</td>
<td>63.61%</td>
<td>65.34%</td>
<td>54.27%</td>
<td>25%/9.38%</td>
</tr></table>

Table 9**:** Quantization **results on W4A4.**

<!-- 25 -->

<!-- **A.3 Case study for QSVD** -->

For case study, we randomly selected examples from the ScienceQA [36] test set, which demonstrate our method's superior performance over QVLM and QASVD baselines.

·As shown in Case 2 and 6, where the FP16 model fails to produce the correct answers, QSVD exhibits more consistent responses between W8A8 and W4A4 settings, and does not contradict the FP16 outputs.

·QSVD also demonstrates surprisingly better performance under low-bit settings in Case 1and 7, where the FP16 model answers incorrectly but QSVD at W4A4 produces the correct response. This may be attributed to the quantization process reducing model noise, as discussed in Dobi-SVD [49], and may partially explain why QSVD sometimes surpasses FP16 accuracy, as observed in Appendix A.2.

·In Case 3, 4,5, and 8, QSVD successfully answers questions that FP16 answers correctly but where other baselines fail under W4A4, highlighting its robustness under extreme quantization.

·Furthermore, in Case 4,5,and 7, other baselines output random tokens or repeat content under 4-bit settings, while QSVD maintains coherent and relevant responses.

<!-- 26 -->

# Case 1: LLaVA-v1.5-7b

**Question:** The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

A.sample B

B. neither; the samples have the same temperature

C.sample A


![](./images/b1c2348a804e2528.jpg)


![](./images/790de38c3141f248.jpg)

**Sample A**

**Sample B**

Mass of each particle:28 u

Mass of each particle:28 u

Average particle speed: 1,800 m/s

Average particle speed: 1,900 m/s

**FP16** **Response:** 1. Compare the average kinetic energies of the particles in each sample. Sample A:The average kinetic energy of the particles in Sample A is 1800000000000000000000000000[truncated]

**QVLM** **W8A8** **Response:** The imageshows two jars filled with green balls, which represent the particles in the gas. The jars are filled with the same number of particles, and the balls are of the same color. The image also provides information about the average kinetic energy of the particles in each jar.

To determine which jar has the higher temperature, wve need to compare the average kinetic energy of the particles in each jar. If the particles in jar A have a higher average kinetic energy, then jar A has a higher temperature. If the particles in jar B have a higher average kinetic energy, then jar B has a higher temperature.

However, the image does not provide enough information to determine the average kinetic energy of the particles in each jar. Therefore, we cannot confidently determine which jar has the higher temperatuire.

**QASVD** **W8A8** **Response**: 1. The diagrams below show two pure samples of gas in identical closed,rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. 2. Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature? A. sample B B. neither; the samples have the same temperature C. sample A

**QSVD** **W8A8** **Response**: 1. The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. 2. Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

A. sample B B. neither; the samples have the same temperature C. sample A

**QSVD W4A4 Response**: A. sample B

Note that **QASVD and QVLM W4A4 output nothing or random number.**

<!-- 27 -->

# Case 2: LLaVA-v1.5-13b

**Question:** Thediagrams below show two pure samples of gas in identical closed,rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?

A.sample B

B. neither; the samples have the same temperature

C.sample A


![](./images/b1c2348a804e2528.jpg)


![](./images/5322084ae5e3d693.jpg)

**Sample A**

**Sample B**

Mass of each particle:28 u

Mass of each particle:28 u

Average particle speed:1,800 m/s

Average particle speed: 1,900 m/s

**FP16** **Response:** The average kinetic energy of the particles in sample A is higher than that of sample B. This is because the particles in sample A are moving faster than those in sample B.The higher average kinetic energy in sample A indicates that the temperature of the gas in sample A is higher than that oof the gas in sample B.

**QVLM** **W8A8** **Response:** The average kinetic energy of the particles in each sample can be determined by calculating the average kinetic energy of each colored ball.The higher the average kinetic energy, the higher the temperature of the sample.

In the image, there are 28 colored balls in each sample. The average kinetic energy of the particles in sample A is calculated as follows:

(1/28) (1/28)(1/28)(1/28) (1/28) (1/28) (1/28) (1/28) (1/28)(1/28)(1/28)(1/28)(1/28)(1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28)(1/28)(1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28)(1/28)(1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28) (1/28)(1/28)(1/28)(1/

**QASVD** **W8A8** **Response:** The average kinetic energy of the particles in sample A is higher than that of sample B. This means that sample A has a higher temperature than sample B.

**QSVD** **W8A8** **Response**: The diagrams show two samples of gas, each in a closed, rigid container. The containers have the same number of colored balls, which represent the gas particles. The samples have different average kinetic energies, which can be determined by comparing the average kinetic energy of the particles in each sample.

Sample A has a higher average kinetic energy than Sample B. This is because the balls in Sample A are moving faster than the balls in Sample B. The average kinetic energy of the particles in Sample A is higher than that of Sample B, which means that Sample A has a higher temperature. **QSVD** **W4A4** **Response:** The average kinetic energy of the particles in sample A is higher than that of sample B.

**Note that QASVD and QVLM W4A4 output nothing or random number.**

<!-- 28 -->

Case 3: LLaVA-v1.5-13b

<table border="1" ><tr>
<td>Question: What is the capital of Louisiana?<br>A.Atlanta<br>B.Frankfort<br>C.New Orleans<br>D.Baton Rouge<br><img src="https://web-api.textin.com/ocr_image/external/c9c57375f70ef947.jpg"></td>
</tr><tr>
<td>FP16 Response: D. Baton Rouge</td>
</tr><tr>
<td>QVLM W8A8 Response: C. New Orleans<br>QVLM W4A4 Response: C. New Orleans<br>QASVD W8A8 Response: C.New Orleans</td>
</tr><tr>
<td>QSVD W8A8 Response: D. Baton Rouge<br>QSVD W4A4 Response: The capital of Louisiana is D. Baton Rouge.</td>
</tr><tr>
<td>Note that QASVD W4A4 output a random number.</td>
</tr></table>

<table border="1" ><tr>
<td>Case 4:LLaVA-v1.5-13b</td>
</tr><tr>
<td>Question: Figure: Kermadec Arc. The Kermadec Arc is a deep sea ecosystem in the southern Pacific Ocean. Which better describes the Kermadec Arc ecosystem?<br>A. It has bright sunlight. It also has organisms that crawl or stick to the ground.<br>B.It has water at the bottom of the ocean. It also has no sunlight.<br>FP16 Response: B. It has water at the bottom of the ocean. It also has no sunlight.<br><img src="https://web-api.textin.com/ocr_image/external/fe2a465c8ec19712.jpg"></td>
</tr><tr>
<td>QVLM W8A8 Response: B. It has water at the bottom of the ocean. It also has no sunlight..<br>QVLM W4A4 Response: B. It has water at the bottom of the ocean. It also has no sunlight..<br>QASVD W8A8 Response: B. It has water at the bottom of the ocean. It also has no sunlight.. QASVD W4A4 Response: The Kermadec Arc is a deep sea ecosystem in the southern Pacific Ocean. Which better describes the Kermadec Arc ecosystem? B. It has water at the bottom of the ocean. It also has no sunlight.</td>
</tr><tr>
<td>QSVD W8A8 Response: B. It has water at the bottom of the ocean. It also has no sunlight.<br>QSVD W4A4 Response: B. It has water at the bottom of the ocean. It also has no sunlight.. Question: The model below represents silver. Silver is a precious metal that is used to make many items, such as jewelry and coins. Complete the statement. Silver is ().<br>A.an elementary substance<br>B.a compound<br>Ag<br>FP16 Response: A. an elementary substance<br><img src="https://web-api.textin.com/ocr_image/external/81904badf0cce523.jpg"></td>
</tr><tr>
<td>QVLM W8A8 Response: A. an elementary substance<br>QVLM W4A4 Response: Silver is a precious metal that is used to make many items, such as jewelry and coins.<br>QASVD W8A8 Response: A. an elementary substance</td>
</tr><tr>
<td>QSVD W8A8 Response: A. an elementary substance<br>QSVD W4A4 Response: A. an elementary substance</td>
</tr><tr>
<td>Note that QASVD W4A4 output a random number.</td>
</tr><tr>
<td>Question: The images below show two pairs of magnets. The magnets in different pairs do not affect each other. All the magnets shown are made of the same material. Think about the magnetic force between the magnets in each pair. Which of the following statements is true?<br>A.The magnetic force is stronger in Pair 1.<br>B. The strength of the magnetic force is the same in both pairs.<br>C. The magnetic force is stronger in Pair 2.<br>Pair 1 Pair 2<br>3- 3-<br>FP16 Response: 1. The magnetic force is stronger in Pair 1.<br>In the image, there are two pairs of magnets, each consisting of two magnets. The magnets in different pairs do not affect each other, and all the magnets are made of the same material. The statement "The magnetic force is stronger in Pair 1" suggests that the magnetic force between the magnets in Pair 1 is stronger than the magnetic force between the magnets in Pair 2. This could be due to various factors such as the orientation, size, or shape of the magnets in each pair. However, without more information or a clear visual comparison, it is not possible to definitively determine which pair has a stronger magnetic force.<br><img src="https://web-api.textin.com/ocr_image/external/12beec6ec3b15eaf.jpg"></td>
</tr><tr>
<td>QVLM W8A8 Response: The magnetic force is stronger in Pair 2.<br>QVLM W4A4 Response: 1. The magnetic force is stronger in Pair1.<br>QASVD W8A8 Response: The magnetic force is stronger in Pair 2.</td>
</tr><tr>
<td>QSVD W8A8 Response: 1. The magnetic force is stronger in Pair 1.<br>In the image, there are two pairs of magnets, each consisting of three magnets. The magnets in each pair are arranged in a straight line, and they are made of the same material. The magnetic force between the magnets in each pair is determined by their strength, size, and arrangement. Since the magnets in Pair 1 are larger and more closely spaced, the magnetic force between them<br>is likely to be stronger than the force between the magnets in Pair 2.<br>QSVD W4A4 Response: The magnetic force is stronger in Pair 1.</td>
</tr><tr>
<td>Note that QASVD W4A4 output a random number.</td>
</tr></table>

<!-- 29 -->

Case 5: LLaVA-v1.5-13b

<!-- 30 -->

Case 7: LLaVA-v1.5-13b

<table border="1" ><tr>
<td>Question: Which ocean is highlighted?<br>A.the Indian Ocean<br>B.the Atlantic Ocean<br>C.the Pacific Ocean<br>D.the Southern Ocean<br>FP16 Response: D. the Southern Ocean<br>QVLM W8A8 Response: D. the Southern Ocean<br>QVLM W4A4 Response: the Southern Ocean<br>QASVD W8A8 Response: C. the Pacific Ocean<br>QASVD W4A4 Response: 2. C. the Pacific Ocean<br><img src="https://web-api.textin.com/ocr_image/external/e1fce7b9ac5b2b8c.jpg"></td>
</tr><tr>
<td>QSVD W8A8 Response: D. the Southern Ocean<br>QSVD W4A4 Response: B. the Atlantic Ocean Question: This organism is Fromia monilis. It is a member oof the animal kingdom. Fromia monilis is commonly called a tile sea star. Did you notice that this tile sea star is missing one of its arms? If a tile sea star loses an arm, it can regrow the arm over time. This ability is called regeneration. Can Fromia monilis cells make their own food?<br>A.yes<br>B.no<br>FP16 Response: 1. Fromia monilis is a type of sea star, also known as a tile sea star.2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food,as they are not capable of photosynthesis.<br><img src="https://web-api.textin.com/ocr_image/external/24c5f18981091d0d.jpg"></td>
</tr><tr>
<td>QVLM W8A8 Response: 1. Fromia monilis is a type of sea star, also known as a tile sea star. 2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food, as they are not capable of photosynthesis.<br>QVLM W4A4 Response: 1. Fromia monilis is a type of sea star that belongs to the animal kingdom. 2. Fromia monilis is commonly known as a tile sea star. 3. The tile sea star is missing one of its arms. 4. Fromia monilis can regrow its arms over time, a process called regeneration. 5. Fromia monilis cells can make their own food.<br>QASVD W8A8 Response: Fromia monilis is a type of sea star, which is an invertebrate. Invertebrates are animals that do not have a backbone. They are typically found in aquatic environments, such as oceans and seas.<br>Sea stars, like Fromia monilis, are known for their ability to regrow lost body parts, such as arms. This process is called regeneration.<br>Sea stars are not capable of making their own food, as they are primarily herbivores.They feed on various types of algae and other plant materials found in their environment.<br>QASVD W4A4 Response: 1.A. Yes</td>
</tr><tr>
<td>QSVD W8A8 Response: 1. Fromia monilis is a type of sea star, also known as a tile sea star.2. The sea star is missing one of its arms, which can regrow over time. 3. Fromia monilis cells cannot make their own food, as they are not capable of photosynthesis.<br>QSVD W4A4 Response: Fromia monilis is a type of sea star that is commonly found in the ocean. It is a member of the animnal kingdom, and it is commonly known as a tile sea star. Fromia monilis has the ability to regenerate lost body parts, such as an arm. This ability is called regeneration.In terms of whether Fromia monilis cells can make their own food, the answer is no. Fromia monilis is a carnivorous organism, which means it relies on other organisms for food. It feeds on small marine animals such as crabs, clams, and other invertebrates.</td>
</tr></table>

<!-- 31 -->

<!-- 32 -->

