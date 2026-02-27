# ProcrustesGPT: Compressing LLMs with Structured Matrices and

# Orthogonal Transformations

**Ekaterina Grishina, Mikhail Gorbunov, Maxim Rakhuba**

HSE University

Correspondence: er.grishina@yandex.ru

# Abstract

Large language models (LLMs) demonstrate impressive results in natural language process-ing tasks but require a significant amount of computational and memory resources. Struc-tured matrix representations are a promnising way for reducing the number of parameters of these models. However, it seems unrealistic to expect that weight matrices of pretrained mod-els can be accurately represented by structured matrices without any fine-tuning. To overcome this issue, we utilize the fact that LLM out-put is invariant under certain orthogonal trans-formations of weight matrices. This insight can be leveraged to identify transformations that significantly improve the compressibility of weights within structuredclasses. The pro-posed approach is applicable to various types of structured matrices that support efficient pro-jection operations. Code is available at: https: //github.com/GrishKate/ProcrustesGPT.

# 1 Introduction

Large language models have achieved remarkable success in language processing and are widely used in a variety of applications, but their deployment is still challenging, as these models hardly fit into a single GPU and require much computational re-sources during the inference and training processes. The research community is actively seeking effi-cient algorithms to reduce the size of pretrained models wvithout sacrificing accuracy.

One approach that has not been fully explored is the use of structured matrices, which can poten-tially not only reduce the number of parameters in the model but also speed up computations. Low-parametric matrix decomposition can be applied directly to the weight matrices, minimizing the difference between the original and decomposed weights,i.e.,\|W-W′\|→minW′∈S,where S is the low-parametric matrix class. Such a factoriza-tion may appear to be a reasonable initial guess for further fine-tuning. Unfortunately, it often leads to high approximation errors for any S when no additional training is performed, as there are no restrictions oon the structure of the weight matrices during pretraining.

To overcome this issue, we utilize invariance of the network output under certain orthogonal trans-formations, which was first observed in (Ashkboos et al., 2024). In particular, for each layer,we aim to find such transformations that lead to the best com-pressibility within the chosen matrix classS. In1our paper, we focus on the sum of Kronecker product representation and GS-matrices (Gorbunov et al., 2024; Dao et al., 2022), but other representations are also possible. Finding an optimal orthogonal transformation is a known linear algebra problem and is called the orthogonal Procrustes problem. The resulting framework is formulated as an opti-mization problem on the weights of the pretrained network, is free from the need for fine-tuning and applicable for different structured matrix represen-tations. In Figure 1, we present relative errors in the Frobenius norm for different layers with and with-out using orthogonal transformations. We observe a noticeable increase in compression ability thanks to optimally chosen orthogonal transformations.

The main contributions of our work:

·We propose a new fine-tuning free framework for compressing LLM weights. The frame-work utilizes orthogonal transformations of the network weight matrices to ensure their compressibility using a low-parametric repre-sentation of choice.

·We formulate this framework as an optimiza-tion problem and develop efficient numerical methods to solve it.

·We conduct experiments with OPT (Zhang et al., 2022) and LLlama2 (Touvron et al., 2023) models.WVe show that in most scenarios, our

<!-- 1 -->

<!-- S707mnfS[TO.so] IA8I870.90sZ:AIXT -->

<table border="1" ><tr>
<td colspan="2">sum of Kronecker products</td>
<td></td>
<td>CS-matrices</td>
<td rowspan="4">ffn w/o Q ffn w/Q<br>attn w/o Q attn w/Q</td>
</tr><tr>
<td rowspan="3">0.5<br>J<br>O<br>J<br>0.4<br>I<br>e<br> <br>e<br>T<br>A<br>0.3<br>e<br>I<br>e<br>J<br>0.2</td>
<td></td>
<td>0.5<br>0.4<br>0.3<br>0.2</td>
<td></td>
</tr><tr>
<td colspan="3">2 4 6 8 10 12 2 4 6 8 10 12</td>
</tr><tr>
<td>layer number</td>
<td></td>
<td>layer number</td>
</tr></table>

Figure 1: Illustration of compressibility of different layers of OPT-125m with and without applying orthogonal transformations Q. We consider two types of structured representations: sum of Kronecker products (Section 4.1) and gS-matrices (Section 4.2). Both representations result in approximately 25% compression for each compressed matrix within the layers.

approach yields more accurate results than alternative fine-tuning-free methods at com-parable compression rates (in the range from 14% to 36%).

# 2 Related work

Existing approaches to neural network compression can be divided into four categories: quantization, distillation, pruning, and matrix factorization. Ma-trix factorization is a promising and relatively un-derexplored technique, and the most widely used approach within this category is a low-rank ap-proximation. The work (Sharma et al., 2023) has shown that by carefully selecting ranks for each of the weights, the performance of LLMs can be improved on some tasks. However, direct applica-tion of SVD to uniform compression of weights leads to poor performance because weights usually have high ranks (Yu and Wu, 2023). Instead,the authors (Wang et al., 2024; Hsu et al., 2022;Yuan et al., 2023; Yu and Wu, 2023; Ji et al., 2024;Chen et al., 2021) use calibration datasets or Fisher in-formation matrix to approximate activations. Other works explore low-rank plus sparse (Li et al., 2023) and low-rank plus quantized decompositions (Saha et al.,2024).

The works (Tahaei et al., 2021; Edalati et al., 2021) have been among the first to apply the Kronecker-product decompositioon for the compres-sion of BERT and GPT2. The paper (Abronin et al., 2024) proposes to enhance the performance of the Kronecker decomposition by adding permutations. These works compress the weight matrices directly and require knowledge distillation or fine-tuning in order to recover performance.

The authors of ModeGPT (Lin et al., 2024) pro-pose to split the weight matrices of the transformer into pairs and jointly compress them using SVD, CR or Nyström approximation. This method is training-free and efficiently preserves model perfor-mance; however, it does not address compression of embedding and head matrices.

Many studies have investigated embedding com-pression, but most of the proposed algorithms re-quire additional training. The works (Xu et al., 2023; Hrinchuk et al., 2020) apply the Tensor Train decomposition to the embedding layer, which of-fers strong compression, but requires training from scratch or hinders model performance. Other meth-ods (Lioutas et al., 2020;Acharya et al., 2019) approximate embedding with low-rank decom-position, but these mnethods require fine-tuning. GroupReduce (Chen et al., 2018) utilizes knowl-edge about words occurrence by weighing tokens with their frequencies and applies block low-rank approximation.

Our goa1 in this study is to maintain model per-formance after weight factorization without addi-tional training. The authors of the prunning ap-proach SliceGPT (Ashkboos et al., 2024) have in-troduced the concept of computational invariance, meaning that orthogonal transformations can be ap-plied to transformer weights without changing the output of the model. SliceGPT uses invariance to project layer activations onto their principal com-ponents and remove columns or rows from weight matrices. In this work, we utilize computational invariance to find a better approximation of the weights with structured representations.

<!-- 2 -->

# 3 Our approach

Our approach consists in iteratively finding orthog-onal transformations Q to obtain optimal com-pression properties of the matrix weights. For brevity, we call these orthogonal transformations "rotations", although they do not necessarily have determinants equal to one. In Section 3.1, we present the concept of rotational invariance.Then, in Section 3.2, we present our approach as an optimization formulation to be solved. NNotably, SliceGPT (Ashkboos et al., 2024) appears to bea particular instance of this formulation, which we also dliscuss further in Section 4.3.

## 3.1 Rotational invariance

In this section, we introduce notation and explain the concept of rotational invariance (Ashkboos et al., 2024). The transformer architecture consists of the repeated series of multi-head self-attention (MHSA) blocks and feed-forward network (FFN) blocks. Between these blocks, there is the Layer-Norm or RMSNorm block. The RMSNorm nor-malizes the input vector:

$$\text {RMSNorm}(x)=\frac {x}{\|x\|_{2}}.$$

LayerNorm is a linear transformation of RMSNorm and a network with LayerNorm can be easily trans-formed to a network with RMSNorm (Ashkboos et al.,2024).

Each of the MHSA and FFN blocks consists of input linear mappings, an activation function, and an output linear mapping. For example, the MHSA block first obtains queries, keys and values (XWq,XWk,XWv)through a linear projection of the inputX. Then these matrices are nonlinearly transformed and multiplied by an output weight matrix((Wo). If we denote the stacked weight ma-trices of the input linear mappings of each block asWinand the weight matrix of the output linear mapping asWout,then we can write:

$$\text {MHSA}(X)=σ\left(X\left[W_{q},W_{k},W_{v}\right]\right)W_{o}+b=\quad =σ\left(XW_{\text {in}}\right)W_{\text {out}}+b,$$

where σ is multi-head attention operation. Simi-larly, the FFN block can be written as

$$\text {FFN}(X)=σ\left(XW_{\text {in}}+b_{\text {in}}\right)W_{\text {out}}+b_{\text {out}},$$

where σdenotes an element-wise activation func-tion,e.g.ReLU.

Let Q be an orthogonal matrix, which is a square matrix satisfyingT=I.It is well known that the Frobenius norm of a matrix\|X\|F2=∑i,jxij2is invariant under orthogonal transformations ofX,i.e.,for any orthogonalQ1andQ2:

$$\left\|Q_{1}XQ_{2}\right\|_{F}=\|X\|_{F}\tag{1}$$

Using this invariance andQTQ=I,we may write

$$\left(\frac {XW_{\text {out}}+X_{\text {skip}}}{\left\|XW_{\text {out}}+X_{\text {skip}}\right\|_{F}}\right)W_{\text {in}}=\quad =\left(\frac {XW_{\text {out}}Q+X_{\text {skip}}Q}{\left\|XW_{\text {out}}Q+X_{\text {skip}}Q\right\|_{F}}\right)\left(Q^{T}W_{\text {in}}\right),\tag{2}$$

whereXskipcomes from the skip connection. As a result,we modify the skip connections to apply Q to the input of the RMSNorm(Xskip)andQTto the output of RMSNorm (the part of (2) in brackets) to keep the model unchanged.

## 3.2 Optimization problem formulation

We aim to improve the compression of the model weights via structured matrices by utilizing rota-tional invariance. In essence, for a given structure of layers, we want to find the rotations that comple-ment well with the chosen structure, then rotate the network and project rotated weights on the struc-tured matrix layers, with as little degradation of per-formance as possible. For each layer,our objective is thus minimizing the L2-difference between the outputs of the rotated network and the outputs of the compressed network on the calibration dataset.

For the ease of presentation and prior to delving into a detailed motivation, let us first articulate the final formulation oof the optimization problem for the l-th layer:

$$\left\|X_{\text {out}}^{\ell }\left(W_{\text {out}}^{\ell }Q_{\ell }-\widehat {W}_{\text {out}}^{\ell }\right)\right\|_{F}^{2}+\tag{3}\quad +λ_{in}\left\|X_{in}^{\ell }\left(W_{in}^{\ell }-Q_{\ell }\widehat {W}_{in}^{\ell }\right)\right\|_{F}^{2}\rightarrow \min _{Q_{\ell }^{T}Q_{\ell }=I,\widehat {W}_{α}^{\ell }\in \mathcal {S}_{α}}\quad α\in \{in,\text {out}\}$$

whereSin andSoutare structured matrix classes that are utilized for compression. Despite the seem-ing simplicity, it is a nontrivial nonconvex opti-mization problem. We propose to approach it by al-ternatingly optimizing between Q\ell and\widehatWi\ell,\widehatWu\ell Individual optimization problems are respectively a Procrustes problem for finding optimal orthogonal matrix and a projection step in a weighted norm for low-parametric representations. Importantly, problems for each layer are independent from each other and can be solved in parallel. We further dis-cuss how we tackle this optimization problem in Section 5.

<!-- 3 -->

<!-- 个 个 个 Q\topQ+ Q\topQ+ Win 0 QT\;Win σ Win 0 Xin Rotate XinQ XinQ Compress XinQ x/l|x|| (Procrustes step) x/||x|| x/l|x|| (projection step) x/I|x|| Xskip XskipQ XskipQ Q-1TQ Q-1TQ Xout Wout Wout Xout WoutQ Xout 0 O 0 Wout inputs inputs inputs -->
![](./images/56522ddb129eebab.jpg)

Figure 2: Illustration of the process of compression of a single transformer layer.

## 3.3 Motivating the optimization formulation

Let us discuss the motivation for (3) with a more general approach. Besides giving us motivation, it also uncovers the connection to the pruning ap-proach of (Ashkboos et al., 2024). As discussed in Section 3.1, by utilizing rotation invariance, one can transform different layers of the network with different rotations without changing the network outputs. However, orthogonal matrices arise in skip connections after the "rotation” step (see Figure 2), and it can be fruitful for future work to additionally compress them.

We denote the weights of the network with the applied set of rotationsQ={Q1,Q2,⋯,QL}as WinQ,\ell,WouQ,\elland the intermediate corresponding inputs asXinQ,\ell,XoutQ,\ellXskipQ,\ell(see Figure 2). Let us denote the input of RMSNNorm as

$$f_{\text {out}}^{\mathcal {Q},\ell }\left(X_{\text {skip}}^{\mathcal {Q},\ell }\right)=X_{\text {out}}^{\mathcal {Q},\ell }W_{\text {out}}^{\mathcal {Q},\ell }+X_{\text {skip}}^{\mathcal {Q},\ell }Q_{\ell -1}^{T}Q_{\ell },$$

and the output of the linear mapping following RMSNorm as

$$f_{in}^{\mathcal {Q},\ell }\left(X_{in}^{\mathcal {Q},\ell }\right)=X_{in}^{\mathcal {Q},\ell }W_{in}^{\mathcal {Q},\ell }$$

For the l -th rotation, objective can be written as

$$\left\|f_{\text {out}}^{\mathcal {Q},\ell }\left(X_{\text {skip}}^{\mathcal {Q},\ell }\right)-\widehat {f}_{\text {out}}^{\ell }\left(X_{\text {skip}}^{\mathcal {Q},\ell }\right)\right\|_{F}^{2}+\quad +λ_{in}\left\|f_{in}^{\mathcal {Q},\ell }\left(X_{in}^{\mathcal {Q},\ell }\right)-\widehat {f}_{in}^{\ell }\left(X_{in}^{\mathcal {Q},\ell }\right)\right\|_{F}^{2}\rightarrow \min _{\substack {\widehat {f}_{in}^{\ell },\widehat {f}_{out}^{\ell },}}$$

Utilization of rotations affects matrices 么Vin\ell,Wc\ell out' while also adding extra matrixQ\ell-1TQ\ellin the skip connection. We aim to compress rotated weights WinQ,\ell,WutQ,\ellwhile also having the possibility of compressing matricesQ\ell-1TQ\ellinto a separate struc-tured matrix\widehatWskip\ellThen, our objective becomes: \|(XoutQ,\ellWoutQ,\ell+XskipQ,\ellQ\ell-1TQ\ell)-. (XoutQ,\ell\widehatWout\ell+XskipQ,\ell\widehatWskip\ell)\|F2+ λin\|XinQ,\ellWinQ,\ell-XinQ,\ell\widehatWin\ell\|F2→ minQ\ellTQ\ell=I,\widehatWα\ell∈Sαα∈{in,out,skip}

Which can be rewritten **as**

$$\|X_{\text {out}}^{\ell }\left(W_{\text {out}}^{\ell }Q_{\ell }-\widehat {W}_{\text {out}}^{\ell }\right)+$$

$$X_{\text {skip}}^{\ell }\left(Q_{\ell }-Q_{\ell -1}\widehat {W}_{\text {skip}}^{\ell }\right)\|_{F}^{2}+$$

$$λ_{in}\left\|X_{in}^{\ell }\left(W_{in}^{\ell }-Q_{\ell }\widehat {W}_{in}^{\ell }\right)\right\|_{F}^{2}\rightarrow \min _{Q_{\ell }^{T}Q_{\ell }=I,\widehat {W}_{α}^{\ell }\in \mathcal {S}_{α}}\quad α\in \{in,\text {out},\text {skip}\}$$

We can approximate this further by abandoning the compression of\widehatWkip\ell,setting it to be equal to Q\ell-1TQ\ell. This way we arrive at (3), and optimiza-tion problem for the e-th layer becomes indepen-dent of the solution for the previous layers. This allows solving problems for different layers in par-allel. Additionally, experiments have shown that balancing the terms in (3) using $λ_{\text {i}}=\frac {\left\|X_{\text {out}}W_{\text {out}}\right\|^{2}}{\left\|X_{\text {i}}W_{\text {i}}\right\|^{2}}$ improves the quality compared toλin=1(see Ta-ble 1).

# 4 Structured matrix representations

In this section, we present different structured ma-trix representations on which we focus in our work. The sum of Kronecker products yields the most consistent accuracy gain within different models. Although the GS-matrix representation resulted in slightly lower accuracy overall, its computation-ally efficient structure offers significant potential to accelerate inference. Finally, we examine matri-ces with zero blocks, revealing connections with pruning techniques (Ashkboos et al.,2024).

<!-- 4 -->

## 4.1 Kronecker products

**Definition 4.1.** Given matricesA∈Rm×nand B∈Rpxq,the Kronecker productA⊗B is the pm×qnblock matrix:

$$A\otimes B=\begin{bmatrix}a_{11}B&\ldots &a_{1n}B\\ :&\cdot .&:\\ a_{m1}B&\ldots &a_{mn}B\end{bmatrix}$$

One Kronecker product is a very restrictive struc-ture, and one usually considers the sum ofr&gt;1Kronecker products for more accurate results. For-tunately,the problem of obtaining the best approx-imation within such a structure (projection opera-tion):

$$\left\|W-\sum _{i=1}^{r}A_{i}\otimes B_{i}\right\|_{F}^{2}\rightarrow \min _{A_{i},B_{i}},\tag{4}$$

has an analytical solution that can be obtained using SVD(Golub and Van Loan, 2013), see Algorithm 1. For better results, we need to use the weighted Frobenius normn (3). The Kronecker product ap-proximation problem with the weight matrix X reads as:

$$\left\|X\left(W-\sum _{i=1}^{r}A_{i}\otimes B_{i}\right)\right\|_{F}^{2}\rightarrow \min _{A_{i},B_{i}}\tag{5}$$

Although it does noot admit a simple solution, the SVD-based solution of (4) can be used for initial-ization for the iterative process. As an iterative procedure, we optimize (5) alternatively with re-spect to{Ai}and{Bi}.Each of the alternating subproblems is solved exactly, details are presented in Appendix A.

### Algorithm 1 SVD-based solution to (4).

**Input**:W∈Rmp×nq,,rank r.

**Output:**{Ai},{Bi}from(4)

Wr=Wrearrange((m p) (nq)→

(m n)(p q))

$$USV^{T}=\text {SVD}\left(W_{r}\right)$$

$$A^{\prime }=U[:,:r]S[:r,:r]^{1/2}$$

$$B^{\prime }=S[:r,:r]^{1/2}V[:,:r]^{T}$$

A=A′.rearrange('(mp) r→rmp')

B=B′..rearrange('(nq)r→rn q')

**return** A,B

## 4.2 gSmatrices

**Definition** **4.2.** GS-matrices are matrices that can be represented in the formPL(LPR)PR,where L,R are block-diagonal matrices and PL,P,PR are permutation matrices.

This class of matrices (Gorbunov et al., 2024) generalizes Monarch (Dao et al., 2022) matrices and describes matrices with low-rank blocks up to a permutation of rows and columns. Thanks to this property, the projection step

$$\left\|W-P_{L}(LPR)P_{R}\right\|_{F}^{2}\rightarrow \min _{L,R}$$

can be performed efficiently using an SVD-based procedure described in (Gorbunov et al., 2024). Likewise for the Kronecker decomposition,the weighted approximation problem

$$\left\|X\left(W-P_{L}(LPR)P_{R}\right)\right\|_{F}^{2}\rightarrow \min _{L,R}$$

does not admit a simple solution. Nevertheless, it can still be solved numerically by alternating iterations with respect to L and R, see Appendix B.

## 4.3 Matrices with zero blocks and relation to SliceGPT

Another structured class one could consider is block-sparse matrices, which include matrices with a single nonzero block. For example, this includes matrices of the forms:

(W 0), \binomW0 - $\left(\begin{array}{cc}W&0\\ 0&0\end{array}\right)$ 

Such structures are not frequently used due to their simplicity and poor expressivity. Nevertheless, when paired with rotations, they can become a use-ful representation. Interestingly enough, we find that utilizing these classes and solving ourobjec-tive have some relation to the SliceGPT method.

**Proposition 4.3.**Letλin=0. LetSinandSskip be matrices with zero columns. Then, solving our objective is equivalent to finding the rotation of the SliceGPT method and column slicing ofWoutand Q\ell-1Q\ell.Row slicingofWinandQ\ell-1Q\ellarises naturally due to the sparse structure of inputs.

Proof. See Appendix C.

☐

<!-- 5 -->

# 5 Optimization algorithm

## 5.1 Orthogonal Procrustes problem

The problem of finding an orthogonal matrix Q,which most closely fits a given matrix A to B, is called an orthogonal Procrustes problem (OPP):

$$\|QA-B\|_{F}\rightarrow \min _{Q^{T}Q=I}$$

It was first solved in the work (Schönemann,1966). The solution isQ=UVT,where UandVTare from the SVD:BAT=UΣVT.

The extension of the Procrustes problem,where an orthogonal matrix Q is multiplied from two sides, is called the weighted orthogonal Procrustes problem (WOPP):

$$\|CQA-B\|_{F}\rightarrow \min _{Q^{T}Q=I}$$

Unfortunately, WOPP does not have a simple an-alytical solution (Lissitz et al., 1976). We solve it by parametrizing Q using the so-called Cayley transform (Golub and Van Loan, 2013)and using conjugate-gradients (see Algorithm 3 and (6)).

## 5.2 Efficient initialization

As we mentioned above, the optimization prob-lem (3) does not admit a simple solution. There-fore, we suggest finding a proper initialization for the arising matrices first. As a good initial point we can use optimal solution in the Frobenius norm:

\|WoutQ-\widehatWout\|F2+\|QTWin-\widehatWin\|F2→min\widehatWout,\widehatWin QTQ=I

To solve this problem, we can rewrite it as:

$$\left\|\left[W_{\text {out}},W_{\text {in}}^{T}\right]Q-\left[\widehat{W}_{\text {out}},\widehat{W}_{\text {in}}^{T}\right]\right\|_{F}^{2}\rightarrow \min _{\substack{\widehat{W}_{\text {out}},\widehat{W}_{\text {in}}\\ Q^{T}Q=I}}$$

which is solved using the alternating scheme called alternating least squares (ALS), see Algorithm 2. In particular, we first find optimal\widehatWou\widehatWinfor the fixed Q,and then optimal orthogonal matrix Q for the fixed\widehatWoutWin(orthogonal Procrustes problem). The process is repeated until the maxi-mum number of iterations is reached.

## 5.3 Alternating iteration

After obtaining initializations for orthogonal lay-ers from ALS in Frobenius norm, we proceed with more computationally challenging weighted opoti-mization scheme, described in Algorithm 3. Even

## Algorithm 2 ALS in the Frobenius norm

**Input:**Wout,Win

SetQ=I.

**for** 1...niters do

Projection step (Section 4):

$$\widehat {W}_{i}=\underset {W}{\arg \min }\left\|Q^{T}W_{i}-W\right\|_{F}^{2}$$

\widehatWout=argmin\|WoutQ-W\|F2W∈Sout 

$$W_{\text {appr}}=\left[\widehat {W}_{\text {out}},\widehat {W}_{\text {in}}^{T}\right]$$

$$W=\left[W_{\text {out}},W_{\text {in}}^{T}\right]$$

Solve OPP, using SVD (Section 5.1)

Q=argmin\|WQ-Wappr\|F2QTQ=I 

**end for**

**return**Q,\widehatWin,\widehatWout

though we only do a handful of steps of the algo-rithm, this step is crucial and noticeably improves results, as is shown in Table 1.

### Algorithm 3 ALS in the weighted norm

Before compression in weighted norm, rotate the network withQinitfrom Algorithm 2.

**Input**Wout,Win,Xout,Xin

SetQ=I

**for** 1...niters do

Weighted norm projection (Appx. A,B):

$$\widehat {W}_{\text {out}}=\underset {W\in \mathcal {S}_{\text {i}}}{\arg \min }\left\|X_{\text {out}}\left(W_{\text {out}}Q-W\right)\right\|_{F}^{2}$$

$\widehat {W}_{in}=\underset {W\in \mathcal {S}}{\arg \min }\left\|X_{in}\left(W_{in}-QW\right)\right\|_{F}^{2}$ W∈So

Solve WOPP by parametrizing Q with Cay-ley transform (6) and using conjugate gradients:

$$+\left\|X_{in}\left(W_{in}-Q\widehat {W}_{in}\right)\right\|_{F}^{2}\quad Q=\underset {Q^{T}Q=I}{\arg \min }\left\|X_{\text {out}}\left(W_{\text {out}}Q-\widehat {W}_{\text {out}}\right)\right\|_{F}^{2}+$$

**end for**

**return**Q,\widehatWin,\widehatWout

#### 5.4 Practical aspects

**Computing** **inputs.** The input of the linear layer X is a matrix of the shapebxn,where b,s and n are respectively the sequence length, the number of calibration samples and the hidden dimension. Typically,b and S are large, so making computa-tions with X is challenging. However, we can use square root of the smallern×ncorrelation matrix XTX∈Rnxn instead ofX to solve the optimiza-tion problem. Indeed,

$$\left\|\left(X^{T}X\right)^{1/2}(\cdots )\right\|_{F}=\|X(\cdots )\|_{F}$$

<!-- 6 -->

To efficiently compute the correlation matrix, we divideX∈Rbsxinto smaller matrices (batches) Xi∈Rbis×n,which fit into memory, and compute XTX=∑iXiTXi

**Embedding and head layers.** For the embed-ding layer the inputsXinare one-hot vectors with ones standing in the position of the chosen tokens. Therefore, the correlation matrixXinTXinis equal to the diagonal matrix D,where Diiis the number of times the i-th token appears in the calibration dataset. This simplifies the weighted problem to:

$$\left\|\sqrt {D}\left(W_{emb}Q-\widehat {W}_{emb}\right)\right\|_{F}^{2}+\|\cdots \|_{F}^{2}\rightarrow \min _{Q}$$

whereWembis the weight of embedding,\widehatWembis its approximation with the matrix decomposition. This prompted us to also experiment with different functions of D and we found that $\sqrt {D+1}$ gave the best results, although 1log(D+1)also worked well (see Tables 1,2). We have also discovered that weighting the model's head with the same diagonal matrix as embedding during compression in Frobe-nius norm additionally improves the performance.

**Orthogonal parametrization.** When rotat-ing the network with the set of weightsQ={Q1,Q2,⋯QL},one downside is additional weightsQ\ell-1TQ\ellthat arise in skip connections and should also be stored as the weights. This nega-tively affects the compression ratio. One trick to reduce the number of parameters is an observation thatQ\ell-1TQ\ellis an orthogonal matrix.

Orthogonal matrices (except for those that have an eigenvalue exactly equal to -1) of sized×d can be represented through the Cayley transform:

$$Q=(I+K)(I-K)^{-1},\tag{6}$$

or through the matrix exponential (if detQ=1):

$$Q=\sum _{n=0}^{\infty }\frac {K^{n}}{n!}\tag{7}$$

where K is skew-symmetric: K=-KT.This allows us to only store the upper triangular part of the matrix K, which reduces the number of param-eters from $d^{2}\text {to}\frac {d(d-1)}{2}$ .If detQ=-1,any rowv of Q can be multipliedby-1to change det to 1. For the Cayley transform, if there exist eigenvectorsui corresponding to -1 eigenvalues, we multiply Q by Householder matricesI-2vvT/\|v\|22,where v=Re(ui)orv=Im(ui) ,to eliminate all -1from the spectrum.

### 6 Experiments

#### 6.1 Setup

We implement our method for OPT (Zhang et al., 2022) and Llama2 (Touvron et al., 2023) models using Hugging Face Transformers (Wolf, 2020). As the calibration data we use 128 sequences of length 2048 from WiKiText2 dataset (Merity et al., 2016). Experiments were run on a single V100GPU with 32GB memory or A100 GPU with 80GB memory.We evaluate zero-shot performance using LM Evaluation Harness (Gao et al., 2024a) across ARC-e,ARC-c (Clark et al., 2018), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2021), and HellaSwag (Zellers et al., 2019) datasets.

#### 6.2 Details of implementation

Every pair of the blocksWout,Winis rotated inde-pendently with its own orthogonal matrix, which allows us to parallelize the computation of the or-thogonal matrices. We have implemented paral-lelization of ALLS in Frobenius norm (Algorithm 2). The speedup depends on the number of processes that can be run simultaneously in GPU memory.

The compression process consists of two stages. Firstly, we compute optimal orthogonal matrices in the Frobenius norm (Algorithm 2),then we ro-tate the network using them and run compression in the weighted norm (Algorithm 3). We use 50ALS iterations in Frobenius norm for small models and 25 iterations for 13b models. We use 1 itera-tion of ALS in the weighted norm for all models. We parametrize orthogonal matrix with the Cayley transform and apply 500 iterations of conjugate gradients to find optimal orthogonal matrices in the weighted norm. We do not compress the Values matrix at all, as it noticeably degrades the results.

#### 6.3 Results

**Generation performance.** We assess the gener-ation performance of the compressed models us-ing the WikiText2 test set. Table 1 compares our method against SliceGPT at approximately 25% compression of the weight matrices. For the de-tails on choice of matrix sizes in decompositions, see Appendix D. The “F” row demonstrates the perplexity achieved by finding optimal orthogonal matrices in the Frobenius norm (see Section 5.2). We observe that compression using the Frobenius norm alone is not sufficient to preserve model per-formance, particularly for small models. However, further compression in the weighted norm helps to

<!-- 7 -->

<table border="1" ><tr>
<td rowspan="3">Method</td>
<td rowspan="3">Norm</td>
<td rowspan="3">Struct.</td>
<td rowspan="3">Coef.</td>
<td colspan="2" rowspan="2">125m</td>
<td colspan="4">OPT</td>
<td colspan="4">Llama2</td>
</tr><tr>
<td colspan="2">2.7b</td>
<td>1</td>
<td>3b</td>
<td colspan="2">7b</td>
<td colspan="2">13b</td>
</tr><tr>
<td>ppl</td>
<td>%</td>
<td>ppl</td>
<td>%</td>
<td>ppl</td>
<td>%</td>
<td>ppl</td>
<td>%</td>
<td>ppl</td>
<td>%</td>
</tr><tr>
<td>Dense</td>
<td></td>
<td></td>
<td></td>
<td>27.65</td>
<td>0</td>
<td>12.47</td>
<td>0</td>
<td>10.13</td>
<td>0</td>
<td>5.47</td>
<td>0</td>
<td>4.88</td>
<td>0</td>
</tr><tr>
<td>SliceGPT</td>
<td></td>
<td></td>
<td></td>
<td>38.65</td>
<td>20.12</td>
<td>14.84</td>
<td>16.51</td>
<td>11.12</td>
<td>16.00</td>
<td>7.60</td>
<td>16.04</td>
<td>6.60</td>
<td>15.94</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>F</td>
<td>Kron.</td>
<td>log(D+1)</td>
<td>58.17</td>
<td>19.59</td>
<td>82.47</td>
<td>15.57</td>
<td>15.20</td>
<td>15.00</td>
<td>16.37</td>
<td>15.04</td>
<td>8.43</td>
<td>14.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>W</td>
<td>Kron.</td>
<td>log(D+1)</td>
<td>38.48</td>
<td>19.59</td>
<td>14.17</td>
<td>15.57</td>
<td>10.87</td>
<td>15.00</td>
<td>8.48</td>
<td>15.04</td>
<td>5.72</td>
<td>14.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>F</td>
<td>Kron.</td>
<td>$\sqrt {D+1}$</td>
<td>55.91</td>
<td>19.59</td>
<td>78.97</td>
<td>15.57</td>
<td>15.38</td>
<td>15.00</td>
<td>11.98</td>
<td>15.04</td>
<td>8.39</td>
<td>14.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>W</td>
<td>Kron.</td>
<td>$\sqrt {D+1}$</td>
<td>36.08</td>
<td>19.59</td>
<td>13.95</td>
<td>15.57</td>
<td>10.67</td>
<td>15.00</td>
<td>6.54</td>
<td>15.04</td>
<td>5.71</td>
<td>14.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>F</td>
<td>GS</td>
<td>$\sqrt {D+1}$</td>
<td>154.09</td>
<td>19.45</td>
<td>152.51</td>
<td>15.57</td>
<td>261.57</td>
<td>15.00</td>
<td>11.65</td>
<td>15.05</td>
<td>8.80</td>
<td>14.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>W</td>
<td>GS</td>
<td>$\sqrt {D+1}$</td>
<td>39.58</td>
<td>19.45</td>
<td>13.81</td>
<td>15.57</td>
<td>10.68</td>
<td>15.00</td>
<td>6.76</td>
<td>15.05</td>
<td>5.96</td>
<td>14.93</td>
</tr></table>

Table 1: Perplexity results on WikiText2. The calibration dataset size is 128 sequennces of 2048 tokens. “Ppl” denotes perplexity,the columns with % show the percentage of parameters compressed. "F" stands for the optimization in Frobenius norm, which is used before optimization in weightednorm, denoted as “W”.“Coef.” stands for the diagonal matrix, which weighs embedding and head. In the rows with llog(D+1) λin=1;in the rows with $\sqrt {D+1},λ_{\text {in}}=\frac {\left\|X_{\text {out}}W_{\text {out}}\right\|^{2}}{\left\|X_{\text {in}}W_{\text {in}}\right\|^{2}}$ 

<table border="1" ><tr>
<td>Model</td>
<td>Method</td>
<td>Struct.</td>
<td>Coef.</td>
<td>ARC-c</td>
<td>ARC-e</td>
<td>HellaS.</td>
<td>PIQA</td>
<td>WinoG.</td>
<td>Average</td>
</tr><tr>
<td rowspan="5">OPT-13b</td>
<td>Dense</td>
<td rowspan="2"></td>
<td rowspan="2"></td>
<td>35.75</td>
<td>61.83</td>
<td>69.88</td>
<td>76.82</td>
<td>65.19</td>
<td>61.89</td>
</tr><tr>
<td>SliceGPT</td>
<td>33.62</td>
<td>61.95</td>
<td>62.99</td>
<td>73.67</td>
<td>63.30</td>
<td>59.10</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>Kron.</td>
<td>log(D+1)</td>
<td>36.52</td>
<td>58.71</td>
<td>68.27</td>
<td>76.22</td>
<td>63.61</td>
<td>60.67</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>Kron.</td>
<td>$\sqrt {D+1}$</td>
<td>36.60</td>
<td>62.25</td>
<td>68.41</td>
<td>76.44</td>
<td>64.96</td>
<td>61.73</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>GS</td>
<td>$\sqrt {D+1}$</td>
<td>35.07</td>
<td>59.09</td>
<td>67.14</td>
<td>76.06</td>
<td>65.59</td>
<td>60.59</td>
</tr><tr>
<td rowspan="5">Llama2-7b</td>
<td>Dense</td>
<td rowspan="3">Kron.</td>
<td rowspan="2"></td>
<td>46.25</td>
<td>74.58</td>
<td>75.99</td>
<td>79.11</td>
<td>68.82</td>
<td>68.95</td>
</tr><tr>
<td>SliceGPT</td>
<td>35.15</td>
<td>56.10</td>
<td>53.04</td>
<td>65.78</td>
<td>62.98</td>
<td>54.61</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>log(D+1)</td>
<td>41.98</td>
<td>68.35</td>
<td>69.72</td>
<td>73.94</td>
<td>67.40</td>
<td>64.28</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>Kron.</td>
<td>$\sqrt {D+1}$</td>
<td>42.32</td>
<td>68.01</td>
<td>70.20</td>
<td>76.39</td>
<td>66.30</td>
<td>64.64</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>GS</td>
<td>$\sqrt {D+1}$</td>
<td>36.69</td>
<td>62.88</td>
<td>66.91</td>
<td>74.37</td>
<td>67.40</td>
<td>61.65</td>
</tr><tr>
<td rowspan="5">Llama2-13b</td>
<td>Dense</td>
<td rowspan="3">Kron.</td>
<td></td>
<td>49.23</td>
<td>77.53</td>
<td>79.36</td>
<td>80.52</td>
<td>72.30</td>
<td>71.79</td>
</tr><tr>
<td>SliceGPT</td>
<td></td>
<td>39.51</td>
<td>62.92</td>
<td>56.98</td>
<td>67.25</td>
<td>67.64</td>
<td>58.86</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>log(D+1)</td>
<td>44.97</td>
<td>73.19</td>
<td>73.43</td>
<td>77.58</td>
<td>70.48</td>
<td>67.93</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>Kron.</td>
<td>$\sqrt {D+1}$</td>
<td>44.80</td>
<td>73.23</td>
<td>73.94</td>
<td>77.53</td>
<td>70.88</td>
<td>68.07</td>
</tr><tr>
<td>ProcrustesGPT</td>
<td>GS</td>
<td>$\sqrt {D+1}$</td>
<td>44.45</td>
<td>73.32</td>
<td>71.19</td>
<td>76.50</td>
<td>70.32</td>
<td>67.16</td>
</tr></table>

Table 2: Zero-shot task performance of compressed OPT-13b,Llama2-7b and Llama2-13b. ProcrustesGPT compresses the weights with Kronecker product. The compression ratio is the same as in Table 1.

regain perplexity, which is shown in the “'W"row. We observe that GS performs consistently worse than Kronecker products in the Frobenius norm for OPT models, and produces approximately the same results in the weighted norm. Our method sur-passes SliceGPT at a similar level of compression. The compression ratios in the tables are presented with respect to the full model size.

Table 3 compares ProcrustesGPT to the methods that do not compress embedding and model's head. SVD-LLM(Wang et al., 2024) applies weighted low-rank approximation, while DISP-LLM (Gao et al., 2024b) and SLEB (Song et al., 2024) are pruning methods. ProcrustesGPT outperforms other baselines at lower compression rates, but its performance starts deteriorating at 36% compres-sion of parameters.

**Zero-shot** **tasks.** We evaluate our models compressed in the weighted norm on five zero-shot tasks. Our method consistently outperforms SliceGPT,as shown in Table 2. The difference is more pronounced for the Llama2 models, where our method surpasses SliceGPT by an average of 9-10%.We notice that compression with sum of Kronecker products better maintains model quality than GS. Table 4 shows that on average Procrustes-GPT achieves better zero-shot performance than other baselines.

<!-- 8 -->

<table border="1" ><tr>
<td rowspan="2">Method</td>
<td colspan="12">Llama2-7b Llama2-13b</td>
</tr><tr>
<td colspan="2">ppl %</td>
<td colspan="2">ppl %</td>
<td colspan="2">ppl %</td>
<td colspan="2">ppl %</td>
<td>ppl</td>
<td>%</td>
<td>ppl</td>
<td>%</td>
</tr><tr>
<td>Dense</td>
<td>5.47</td>
<td>0</td>
<td>5.47</td>
<td>0</td>
<td>5.47</td>
<td>0</td>
<td>4.88</td>
<td>0</td>
<td>4.88</td>
<td>0</td>
<td>4.88</td>
<td>0</td>
</tr><tr>
<td>SVD-LLM</td>
<td>7.86</td>
<td>14.44</td>
<td>9.73</td>
<td>25.00</td>
<td>14.39</td>
<td>35.58</td>
<td>6.34</td>
<td>14.64</td>
<td>7.53</td>
<td>25.36</td>
<td>10.08</td>
<td>36.09</td>
</tr><tr>
<td>DISP-LLM</td>
<td>6.80</td>
<td>14.31</td>
<td>8.52</td>
<td>25.02</td>
<td>10.92</td>
<td>35.60</td>
<td>6.23</td>
<td> 14.60</td>
<td>7.90</td>
<td>25.36</td>
<td>10.05</td>
<td>36.13</td>
</tr><tr>
<td>SLEB</td>
<td>6.95</td>
<td>12.01</td>
<td>10.39</td>
<td>24.03</td>
<td>22.76</td>
<td>36.04</td>
<td>5.85</td>
<td>12.19</td>
<td>7.73</td>
<td>24.37</td>
<td>11.36</td>
<td>36.56</td>
</tr><tr>
<td>ProcrustesGPT (Kron)</td>
<td>6.43</td>
<td>14.07</td>
<td>8.19</td>
<td>25.09</td>
<td>19.55</td>
<td>36.11</td>
<td>5.68</td>
<td>14.30</td>
<td>6.95</td>
<td>25.48</td>
<td>16.88</td>
<td>36.66</td>
</tr><tr>
<td>ProcrustesGPT (GS)</td>
<td>6.65</td>
<td>14.08</td>
<td>7.97</td>
<td>25.08</td>
<td>14.20</td>
<td>36.12</td>
<td>5.94</td>
<td>14.30</td>
<td>7.02</td>
<td>25.48</td>
<td>10.85</td>
<td>36.66</td>
</tr></table>

Table 3: Perplexity of compressed Llama2 on WikiText2. Embedding and head are not compressed for all methods. The calibration dataset size is 128 sequences of 2048 tokens. % shows the percentage of parameters compressed.

<table border="1" ><tr>
<td>Method</td>
<td>%</td>
<td>ARC-c</td>
<td>ARC-e</td>
<td>HellaS.</td>
<td>PIQA</td>
<td>WinoG.</td>
<td>Average</td>
</tr><tr>
<td>Dense</td>
<td>0</td>
<td>49.23</td>
<td>77.53</td>
<td>79.36</td>
<td>80.52</td>
<td>72.30</td>
<td>71.79</td>
</tr><tr>
<td>SVD-LLM</td>
<td>14.64</td>
<td>39.25</td>
<td>65.61</td>
<td>63.92</td>
<td>73.83</td>
<td>68.35</td>
<td>62.19</td>
</tr><tr>
<td>DISP-LLM</td>
<td>14.60</td>
<td>47.61</td>
<td>70.12</td>
<td>74.77</td>
<td>76.93</td>
<td>69.61</td>
<td>67.81</td>
</tr><tr>
<td>SLEB</td>
<td>12.19</td>
<td>46.33</td>
<td>72.77</td>
<td>74.11</td>
<td>78.18</td>
<td>69.85</td>
<td>68.25</td>
</tr><tr>
<td>ProcrustesGPT (Kron)</td>
<td>14.30</td>
<td>45.56</td>
<td>74.16</td>
<td>74.29</td>
<td>77.58</td>
<td>70.24</td>
<td>68.37</td>
</tr><tr>
<td>ProcrustesGPT (GS)</td>
<td>14.30</td>
<td>45.05</td>
<td>73.40</td>
<td>71.71</td>
<td>76.77</td>
<td>70.56</td>
<td>67.50</td>
</tr><tr>
<td>SVD-LLM</td>
<td>25.36</td>
<td>32.76</td>
<td>54.34</td>
<td>54.19</td>
<td>68.12</td>
<td>65.98</td>
<td>55.08</td>
</tr><tr>
<td>DISP-LLM</td>
<td>25.36</td>
<td>40.27</td>
<td>61.83</td>
<td>69.40</td>
<td>73.56</td>
<td>63.30</td>
<td>61.67</td>
</tr><tr>
<td>SLEB</td>
<td>24.37</td>
<td>38.14</td>
<td>63.47</td>
<td>66.78</td>
<td>76.39</td>
<td>60.70</td>
<td>61.10</td>
</tr><tr>
<td>ProcrustesGPT (Kron)</td>
<td>25.48</td>
<td>38.48</td>
<td>70.03</td>
<td>66.42</td>
<td>75.03</td>
<td>66.06</td>
<td>63.20</td>
</tr><tr>
<td>ProcrustesGPT (GS)</td>
<td>25.48</td>
<td>41.04</td>
<td>71.38</td>
<td>66.12</td>
<td>74.71</td>
<td>67.17</td>
<td>64.10</td>
</tr><tr>
<td>SVD-LLM</td>
<td>36.09</td>
<td>26.96</td>
<td>43.22</td>
<td>43.35</td>
<td>61.53</td>
<td>60.54</td>
<td>47.12</td>
</tr><tr>
<td>DISP-LLM</td>
<td>36.13</td>
<td>30.72</td>
<td>53.03</td>
<td>60.65</td>
<td>68.72</td>
<td>58.64</td>
<td>54.35</td>
</tr><tr>
<td>SLEB</td>
<td>36.56</td>
<td>33.62</td>
<td>52.15</td>
<td>58.42</td>
<td>71.27</td>
<td>59.67</td>
<td>55.03</td>
</tr><tr>
<td>ProcrustesGPT (Kron)</td>
<td>36.66</td>
<td>29.69</td>
<td>52.95</td>
<td>48.56</td>
<td>64.80</td>
<td>59.98</td>
<td>51.20</td>
</tr><tr>
<td>ProcrustesGPT (GS)</td>
<td>36.66</td>
<td>33.70</td>
<td>61.87</td>
<td>52.58</td>
<td>67.85</td>
<td>62.43</td>
<td>55.69</td>
</tr></table>

Table 4: Zero-shot performance of compressed Llama2-13b. Embedding and head are not compressed for all methods. % shows the percentage of model parameters compressed.

### 7 Conclusion

This paper presents an approach to LLM compres-sion with structured matrix factorizations, which is suitable for compression with various types of decompositions, including Kronecker products and gS matrices. Our method maintains performance in generation and zero-shot tasks, and does not re-quire recovery fine-tuning. We hope this work will inspire further research on training-free compres-sion with structured representations.

### 8 Limitations

A natural question may arise if the models com-pressed using ProcrustesGPT can be fine-tuned with standard methods such as LoRA. The struc-tured weights may not align well with the low-rank nature of adapters. As a result, after the fine-tuning we will need to store the structured representation of initial weights and the LoRA adapters separately, which may be inconvenient. Therefore, models compressed using structured factorizations require the development of PEFT methods that are better suited to these structures.

### 9 Acknowledgments

Support from the Basic Research Program of HSE University is gratefully acknowledged. The calcu-lations were performed in part through the compu-tational resources of HPC facilities at HSE Univer-sity(Kostenetskiy et al., 2021).

<!-- 9 -->

## References

V Abronin, A Naumov, D Mazur, D Bystrov, K Tsarova, Ar Melnikov, I Oseledets, Sergey Dolgov,R Brasher, and Michael Perelshtein. 2024. Tqcompressor: improving tensor decomposition methods in neu-ral networks via permutations. arXiv preprint arXiv:2401.16367.

Anish Acharya, Rahul Goel, Angeliki Metallinou,and Inderjit Dhillon. 2019. Online embedding compres-sion for text classification using low rank matrix fac-torization. In Proceedings of the aaai conference on artificial intelligence, volume 33, pages 6196-6203.

Saleh Ashkboos, Maximilian L Croci, Marcelo Gen-nari do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024.

Yonatan Bisk, Rowan Zellers,Jianfeng Gao,Yejin Choi, et al. 2020. Piqa: Reasoning about physical com-monsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432-7439.

Patrick Chen, Si Si, Yang Li, Ciprian Chelba, and Cho-Jui Hsieh.2018. Groupreduce: Block-wise low-rank approximation for neural language model shrinking. Advances in Neural Information Processing Systems, 31.

Patrick Chen, Hsiang-Fu Yu, Inderjit Dhillon, and Cho-Jui Hsieh. 2021. Drone: Data-aware low-rank com-pression for large nlp models. Advances in neural information processing systems, 34:29321-29334.

Peter Clark,Isaac Cowhey,Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question an-swering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Tri Dao, Beidi Chen, Nimit S Sohoni, Arjun De-sai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Ré. 2022. Monarch: Expressive structured matrices for efficient and accurate training. In International Conference on Machine Learning, pages 4690-4721. PMLR.

Ali Edalati, Marzieh Tahaei, Ahmad Rashid, Vahid Par-tovi Nia, James J Clark,and Mehdi Rezagholizadeh. 2021. Kronecker decomposition for gpt compression. arXiv preprint arXiv:2110.08152.

Leo Gao,Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds,Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang,An-ish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024a. A framework for few-shot language model evaluation.

Shangqian Gao,Chi-Heng Lin,Ting Hua,Zheng Tang, Yilin Shen, Hongxia Jin,and Yen-Chang Hsu.2024b. Disp-llm: Dimension-independent structural prun-ing for large language models. Advances in Neural Information Processing Systems, 37:72219-72244.

Gene H Golub and Charles F Van Loan. 2013. Matrix computations. JHU press.

Mikhail Gorbunov, Nikolay Yudin, Vera Soboleva, Aibek Alanov, Alexey Naumov, and Maxim Rakhuba. 2024. Group and shuffle: Efficient structured or-thogonal parametrization. In Advances in Neural Information Processing Systems, volume 37, pages 68713-68739.Curran Associates,Inc.

Oleksii Hrinchuk, Valentin Khrulkov, Leyla Mir-vakhabova, Elena Orlova, and Ivan Oseledets. 2020. Tensorized embedding layers. In Findings of the As-sociation for Computational Linguistics: EMNLP 2020,pages 4847-4860.

Yen-Chang Hsu, Ting Hua, Sungen Chang, Qian Lou, Yilin Shen, and Hongxia Jin. 2022. Language model compression with weighted low-rank factorization. arXiv preprint arXiv:2207.00112.

Yixin Ji, Yang Xiang, Juntao Li, Qingrong Xia,Zi Ye, Xinyu Duan,Zhefeng Wang, Kehai Chen, and Min Zhang. 2024. Adaptive feature-based low-rank com-pression of large language models via bayesian op-timization. In Findings of the Association for Com-putational Linguistics: EMNLP 2024, pages 4152-4168.

PS Kostenetskiy, RA Chulkevich, and VI Kozyrev. 2021. HPC resources of the higher school of economics. In Journal of Physics: Conference Series, volume 1740, page 012050.

Yixiao Li, Yifan Yu, Qingru Zhang,Chen Liang, Pengcheng He, Weizhu Chen,and Tuo Zhao.2023. Losparse: Structured compression of large language models based on low-rank and sparse approximation. In International Conference on Machine Learning, pages 20336-20350. PMLR.

Chi-Heng Lin, Shangqian Gao, James Seale Smith, Ab-hishek Patel,Shikhar Tuli, Yilin Shen, Hongxia Jin, and Yen-Chang Hsu. 2024. Modegpt: Modular de-composition for large language model compression. arXiv preprint arXiv:2408.09632.

Vasileios Lioutas, Ahmad Rashid, Krtin Kumar, Md Ak-mal Haidar, and Mehdi Rezagholizadeh. 2020. Im-proving word embedding factorization for compres-sion using distilled nonlinear neural decomposition. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2774-2784.

Robert W Lissitz, Peter H Schönemann, and James C Lingoes. 1976. A solution to the weighted procrustes problem inwhich the transformation is in agreement with the loss function. Psychometrika, 41:547-550.Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture mod-els.arXiv preprint arXiv:1609.07843.

<!-- 10 -->

Rajarshi Saha, Naomi Sagan, Varun Srivastava, An-drea J Goldsmith, and Mert Pilanci. 2024. Com-pressing large language models using low rank and low precision decomposition. arXiv preprint arXiv:2405.18886.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat-ula,and Yejin Choi. 2021. Winogrande: An adver-sarial winograd schema challenge at scale. Commu-nications of the ACM, 64(9):99-106.

Peter H Schönemann. 1966. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1-10.

Pratyusha Sharma, Jordan T Ash, and Dipendra Misra. 2023.The truth is in there: Improving reasoning in language models with layer-selective rank reduction. arXiv preprint arXiv:2312.13558.

Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim. 2024. Sleb: Streamlining llms through redundancy verification and elimination of transformer blocks. arXiv preprint arXiv:2402.09025.

Marzieh S Tahaei, Ella Charlaix, Vahid Partovi Nia, Ali Ghodsi,and Mehdi Rezagholizadeh. 2021. Kro-neckerbert: Learning kronecker decomposition for pre-trained language models via knowledge distilla-tion. arXiv preprint arXiv:2109.06243.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and effi-cient foundation language modlels. arXiv preprint arXiv:2302.13971.

Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. 2024. Svd-llm: Truncation-aware singular value de-composition for large language model compression. arXiv preprint arXiv:2403.07378.

Thomas Wolf. 2020. Transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771.

Mingxue Xu, Yao Lei Xu, and Danilo P Mandic. 2023. Tensorgpt: Efficient compression of the embedding layer in llms based on the tensor-train decomposition. arXiv preprint arXiv:2307.00526.

Hao Yu and Jianxin Wu. 2023. Compressing transform-ers: features are low-rank, but weights are not! In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11007-11015.

Zhihang Yuan,Yuzhang Shang, Yue Song, Qiang Wu,Yan Yan, and Guangyu Sun. 2023. Asvd: Activation-aware singular value decomposition for compressing large language models. arXiv preprint arXiv:2312.05821.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe,Moya Chen, Shuohui Chen, Christopher De-wan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

## A Kronecker approximation in weighted norm

LetA∈Rr×m1×n1 B∈Rrxm2xn2. The problem of Kronecker approximation in the weighted norm can be written as

$$g(A,B)=\left\|Y-X\sum _{i}^{r}\left(A_{i}\otimes B_{i}\right)\right\|_{F}^{2}\rightarrow \min _{A,B}$$

We can not solve this problem explicitly, so we will iteratively optimize this expression with respect to A for the fixed B and with respect to B for the fixed A. We can rewrite it as

$$g(A,B)=\text {tr}\left(Y^{T}Y-2Y^{T}X\left(\sum _{i}^{r}A_{i}\otimes B_{i}\right)+\right.\quad \left.+\left(\sum _{i}^{r}A_{i}\otimes B_{i}\right)^{T}X^{T}X\left(\sum _{j}^{r}A_{j}\otimes B_{j}\right)\right)$$

Now we take the differential with respect to A

$$dg=\text {tr}\left(2Y^{T}X\left(\sum _{i}^{r}dA_{i}\otimes B_{i}\right)+\right.$$

$$\left.+2\left(\sum _{i}^{r}A_{i}\otimes B_{i}\right)^{T}X^{T}X\left(\sum _{j}^{r}dA_{j}\otimes B_{j}\right)\right)$$

To find the optimal A we should equate the gradient with respect to A to zero. It is complicated to write the solution using formulas, so we will use tensor diagrams instead. In tensor diagrams, the circles denote multidimensional arrays, the connections between them denote the summation by dimen-sions. Let us equate the dg to zero and illustrate it as follows:

$$\text {tr}\left(\left(\sum _{i}^{r}A_{i}\otimes B_{i}\right)^{T}X^{T}X\left(\sum _{j}^{r}dA_{j}\otimes B_{j}\right)\right)\quad -\text {tr}\left(Y^{T}X\left(\sum _{i}^{r}dA_{i}\otimes B_{i}\right)\right)=0$$

<!-- 11 -->

<!-- A n1 dA r n22 n2 m1 B B m1 n1 $\textcircled {8}$ m1 =0 n2 m2 m2 m2 XTX YTX -->
![](./images/68c36a2c76589698.jpg)

Now we equate the gradient with respect to A to zero:

<!-- A n1 n1 r r r r n2 m1 m1 B B B m1 m1 n1B m1 m2 m2 XTX YTX -->
![](./images/cecac9850747ed8e.jpg)

We can reshape the tensors in the left and right parts into matrices. Let us denote the matrix in the right part by D. The left part is the tensor A reshaped into matrix of sizerm1n1multiplied by a matrix C:

<!-- m1 C rm1A $A)^{\underline {n_{1}}}={m_{1}}$ D )n1 -->
![](./images/d9a45fc8bb0f4218.jpg)

Then the solution is

$$A=\left(C^{+}D\right)\text {.reshape}\left(r,m_{1},n_{1}\right)$$

The pseudo-code is shown in Algorithm 4.

## Algorithm 4 Kronecker appr. in weighted norm

**Input**B∈Rr×m2×n2,Y∈Rm1m2×n1n2,X∈Rm1m2xm1m2

**Solve**\|Y-X∑ir(Ai⊗Bi)\|F2→minA

$$Z=X^{T}X\text {.reshape}\left(m_{1},m_{2},m_{1},m_{2}\right)$$

C=einsum('rij,nkj,aibk→ranb',B,B,Z)

$$C=C\text {.reshape}\left(rm_{1},rm_{1}\right)$$

$$T=X^{T}Y\text {.reshape}\left(m_{1},m_{2},n_{1},n_{2}\right)$$

$$D=\text {einsum}\left(\text {'rij,aibj}\rightarrow \text {rab'},B,T\right)$$

D=D.reshape (rm1,n1)

$$A=\left(C^{+}D\right)\text {.reshape}\left(r,m_{1},n_{1}\right)$$

**return** A

## B GSpproximation in weighted norm

Let L be block-diagonal matrix with kl blocks of sizei∈Rb1b2,R be block-diagonal matrix with kr blocks of size i∈br1×br2.Let us solve the approximation with GS matrices with fixed permutationsPL,P,PRin the weighted norm:

$$\left\|Y^{\prime }-X^{\prime }P_{L}LPRP_{R}\right\|_{F}^{2}\rightarrow \min _{L,R}$$

Due to the unitary invariance of Frobenius norm and orthogonality of permutation matrices, we can rewrite it as

$$\|Y-XLPR\|_{F}^{2}\rightarrow \min _{L,R},$$

whereX=X′PL,Y=Y′PRT Wecan not solve this problem analytically, so we will optimize itera-tively with respect to Land R.

To find optimal R for the fixed L we have to solve the least squares problem for each of the blocksRi. Let us divide Y and XLP by columns into kr blocks Yi∈Rkl·bl1xbl2and(XLP)i∈Rkl·bl1xbl2. Then we can find the optimalR as

$$R_{i}=(XLP)_{i}^{+}Y_{i}$$

To solve the optimization problem with respect to L for the fixed R, we use built-in iterative algorithm for least squares problem. However, to avoid poorly conditioned systems, we first divide X and PR into kl blocks by columns and rows respectively, and make QR decomposition ofXiand(PR)i.The result is shown in Algorithm 5 below.

<table border="1" ><tr>
<td>lgorithm 5 Find optimal L</td>
</tr><tr>
<td>Input Xi∈ Rkl⋅bl1×bl1,Y E RRkl·bl1xkr·br2,(PR)i∈Rkl×kr⋅br2<br>Solve\|Y-∑iklXiLi(PR)i\|F2→inL<br>QXi,RXi=qr(Xi)<br>Q(PR)iT,R(PR)iT=qr((PR)iT)<br>L=argminL(Y-∑iklQXiLiQ(PR)iTT)<br>Li=RXi+Li(R(PR)iTT)+<br>return L</td>
</tr></table>

## C Proof of Proposition 4.3

Firstly,let us write

$$\left\|X_{\text {out}}\left(W_{\text {out}}Q-\widehat {W}_{\text {out}}\right)+\right.$$

$$+X_{\text {skip}}\left(Q-Q_{\ell -1}\widehat {W}_{\text {skip}}\right)\|_{F}^{2}=$$

$$=\|\left(X_{\text {out}}W_{\text {out}}+X_{\text {skip}}\right)-$$

$$-\left(X_{\text {out}}\widehat {W}_{\text {out}}+X_{\text {skip}}Q_{\ell -1}\widehat {W}_{\text {skip}}\right)Q^{T}\|_{F}^{2}$$

Note that slicing of the columns of matrices \widehatWkip and Wout make matrix Xou\widehatWouQT+ XskipQ\ell-1\widehatWskipQT at most of rank d, where d is the dimension after slicing.If we now show, that we can choose appropriate matrices Q\widehatWou\widehatWskipsuch that they correspond to best

<!-- 12 -->

low-rank approximation ofXoutWout+Xskip,we will show that these matrices are the analytical solu-tions to our problem. Let P equal to the following projector:

$$P=\left(\begin{array}{cc}I_{d}&0\\ 0&0\end{array}\right)$$

Consider following matrices for\widehatWout,\widehatWskip:

\widehatWout=WoutQP; \widehatWskip=Q\ell-1TQP

Substituting, we get:

$$\left\|\left(X_{\text {out}}W_{\text {out}}+X_{\text {skip}}\right)Q(I-P)\right\|_{F}^{2}\rightarrow \min _{Q^{T}Q=I}$$

Solution to which is well-known and revolves around computing SVD of(XoutWout+Xskip) (taking principal components). This also corre-sponds to the best low-rank approximation and therefore found solution is optimal.

Up to a normalization layer, this corresponds to the rotations and slices ofWutandQ\ell-1TQ\ell layers applied in SliceGPT. Sparse block structure that arises in inputs after the first layers allows to also sliceWinandQ\ell-1TQ\ellalong other dimension, making it equivalent to full SliceGPT scheme.

## D Choice of sizes for factorizations.

To compress a matrix M of sizen×mat approxi-mately p/q of its parameters with the sum of r Kro-necker products∑irAi⊗Bi,,a natural choice is to setr=p,Aiof sizeq×1andBiof sizen/qxm We assume n is divisible by q. Then the number of parameters in the sum∑ir=pAi⊗Biequals p(q+(n/q)m)=qp+nm(p/q),which is ap-proximately p/q of initial matrix size. In our case, we useq=4andr=3,,yielding \;25% com-pression rate for layers. We useq=5,r=8for \;37.5% compression,q=1,r=2for250%compression.

It is important to note, that splitting by q should be applied on the side that is multiplied by an or-thogonal mmatrix, so that it has an effect on approxi-mation error. For example, if the matrixWinis mul-tiplied by orthogonal matrix from the left, then de-composition should beAi∈Rq1Bi∈R/q orAi∈R/q×m,Bi∈Rq×1.IfWoutis multiplied from the right,thenAi∈R1xq,Bi∈Rnxm/qor Ai∈Rn×/q,Bi∈R1×q

When compressing the matrix M $\underline {M}\quad \in$ Rkl*bl1kr*br2 with GS decomposition using L with kl blocks of the sizebl1×bl2and R with kr blocks of size br1×br2,the compression ratio equals $c=\frac {kl*bl_{1}*bl_{2}+kr*br_{1}*br_{2}}{kl*kr*bl_{1}*br_{2}}$ .From the definition of GSmatrices,kl*bl2=kr*br1, which means that it is easy to findbl2 and br1 if c,kl,bl1,kr,br2are known. To compress square matrices atc=3/4,we choosekl,kr=4,2.For rectangular matrices,kl,k=4,8..For embedding and headkl,kr=1,4.

<!-- 13 -->

