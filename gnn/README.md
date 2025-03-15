
# Graph Neural Networks 


## Graph Convolutional Networks 
A standard convolution in Deep Learnin such as in Convolutional Neural
Networks applies a filter to a signal (image). However in discrete 
graph structures, there is no natural grid-like structure 
like in images, so we need a different way to define convolutions.

A spectral convolution on a graph is a mathematical way to filter 
a signal defined on a graph using the eigenvalues and eigenvectors 
of a special matrix called the graph Laplacian.


### Message and Aggregate View
$$
h_{v}^{(l)} = \sigma(\Sigma_{u \in N(v)}W^{(l)}\frac{1}{\sqrt{|N(u)||N(v)|}}h_{u}^{l-1})
$$

where: 

- $h_v^{(l)}$ is the node feature representation of node $v$ at layer $$ l $$.
- $N(v)$ represents the **neighbors** of node $ v $.
- $W^{(l)}$ is the trainable weight matrix at layer $ l $.
- $\frac{1}{\sqrt{|N(u)| |N(v)|}}$ is a **normalization term** that accounts for the degree of nodes.
- $h_u^{(l-1)}$ represents the feature vector of neighboring node $ u $ from the previous layer $l - 1$.
- $\sigma(\cdot)$ is a **non-linear activation function** (e.g., ReLU).

Is $v$ part of $N(v)$ : for this paper, yes since there are self loops, $\tilde{A} = A + I$


### The Graph Laplacian 
A graph Laplacian, $L$ is a a matrix that helps us analyze graph structures. The Laplacian matrix is useful because its eigenvalues and eigenvectors provide important structural information about the graph. There are different forms of Laplacians, but in this paper the normalized graph Laplacian is used:

$$
L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
$$

where 
- $I_N$ is the identity matrix
- $D$ is the degree matrix 
- $A$ is the adjacency matrix 

The graph Fourier transform is the process of transforming a signal on the graph into the spectral domain using the eigenvectors of $L$. the eigendecomposition of L is:

$$
L = U \Lambda U^{T}
$$

where 
- $U$ is the matrix of eigenvectors
- $\Lambda$ is the diagonal matrix of eigenvalues

### Spectral Graph Convolution Defn:
We can define spectral convolutions as the multiplication of a signal $x$ by a filter function $g{\theta}$ in the Fourier domain. 

$$
g_{\theta} * x = Ug_{\theta}U^Tx
$$

where 

- $x$: A signal on the graph (feature for each node, could be a representation vector)
- $g_{\theta}$: A filter applied in the Fourier domain
- $U$: The matrix of eigenvectors of the normalized graph Laplacian
- $U^{T}x$: Graph Fourier transform of x
- $g_{\theta}U^{T}x$: Applies filter in the Fourier domain
- $Ug_{\theta}U^{T}x$ Inverse Fourir transform to convert back to node space. 

However, computing this directly is computationally expensive because:
- It requires computing the full eigendecomposition of $L$ which takes $O(N^2)$ operations.
- Multiplication with U can be costly for large graphs. 

### Spectral Graph Convolution Approximation using Chebyshev polynomials:
To make spectral convolutions computationally feasible, we approximate the filter function $g_\theta(\Lambda)$ using Chebyshev polynomials, which are special polynomials that help approximate functions efficiently. 

$$
g_\theta(\Lambda) \approx \Sigma_{k=0}^{K}\theta_{k}'T_{k}(\tilde{\Lambda})
$$

where:
- $\tilde{\Lambda} = \frac{2}{\lambda_{max}}\Lambda - I_{N}$ is rescaled version of $\Lambda$
- $\theta_{k}'$ are Cheybyshev coefficients, which are parameters that we learn.
- The  Chebyshev polynomials are defined recursively as:
  
$$
T_k(x)=2xT_{k-1}(x) - T_{k-2}(x)
$$

where $T_0(x) = 1$ and $T_1(x) = x$

Since $\tilde{\Lambda}$ is defined via $\tilde{L}$, the convolution operation follows:

$$
g_{\theta'}  * x \approx \Sigma_{k=0}^{K}\theta_{k}'T_{k}(\tilde{L})x
$$

where $\tilde{L} = \frac{2}{\lambda_{max}}L-I_N$ is the rescaled Laplacian.

This is useful as instead of computing the full eigendecomposition of $L$, we can:
1. Use the rescaled Laplacian $\tilde{L}$
2. Compute the Chebyshev polynomooisl $T_k{\tilde{L}}$ recursively.
3. Apply the convolution efficiently ins $O(|E|) time. 

This bypasses the expensive Fourier transform and allows graph convolutions to be computed locally.


### Graph Convolutional Networks 
A Graph Convolutional Network (GCN) is built by stacking multiple graph convolution layers, followed by non-linear activation functions. The idea is that each layer aggregates information from neighboring nodes, similar to how convolutional layers work in CNNs for images. Previously we defined the general form for spectral graph convolutions as:

$$
g_{\theta'}  * x \approx \Sigma_{k=0}^{K}\theta_{k}'T_{k}(\tilde{L})x
$$

This applies a K-th order Cheybyshev polynomial expansion, which makes it computational feasible. Now what if we set K=1? Setting K=1, the filter functions simplifies to

$$
g_{\theta'} * x \approx \theta_{0}'x + \theta_{1}'(L-I_N)x
$$

Expanding $L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$, we get:

$$
g_{\theta'} * x = \theta_{0}'x - \theta_{1}'D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$

Instead of two parameters $\theta_{0}'$ and $\theta_{1}'$, we merge them into a single parameter $\theta$. 

$$
g_{\theta'} * x \approx \theta(I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
$$

The issue now is that $I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ can lead to exploding or vanishing gradients when stacking multiple layers (some eigenvalue reason). To fix this we apply the renormalization trick: 

$$
I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \rarr \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}
$$

$$
g_{\theta'} * x \approx \theta( \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})x
$$

where 
- $\tilde{A} = A + I_{N}$ (adds self-loops the graph)
- $\tilde{D_{ii}} = \Sigma_j{\tilde{A}_{ij}}$ (degree matrix for $\tilde{A}$ )

This modification ensures that 
- The eigenvalues remain bounded between $[0,2]$ preventing instability
- Each node includes its own features in addition to its neigbors

### Extending to mult-feature graphs 
So far, we've considered a single feature per node. But real-world graphs (e.g., social networks, molecular graphs) often have multiple features per node. To generalize to multiple input channels, we represent the node features as a matrix $X \in \R^{N \times C}$ (feature matrix of all the nodes) where 
- $N$ is the number of nodes in the graph 
- $C$ is the number of input features per node. 

$$
g_\theta \star X \approx \tilde{D}^{- \frac{1}{2}} \tilde{A} \tilde{D}^{- \frac{1}{2}} X \Theta
$$

### GCN Propagation Rule

$$
H^{(l+1)} = \sigma \left( \tilde{D}^{- \frac{1}{2}} \tilde{A} \tilde{D}^{- \frac{1}{2}} H^{(l)} W^{(l)} \right)
$$

where:
- $\tilde{A} = A + I_N$ is the adjacency matrix of the undirected graph $\mathcal{G}$ with added self-connections.
- $I_N$ is the identity matrix.
- $\tilde{D_{ii}} = \sum_j \tilde{A}_{ij}$ is the degree matrix corresponding to $ \tilde{A} $.
- $W^{(l)}$ is a layer-specific trainable weight matrix.
- $\sigma(\cdot)$ denotes an activation function, such as ReLU: $ \text{ReLU}(x) = \max(0, x) $.
- $H^{(0)} = X$ is the input feature matrix of shape $ \mathbb{R}^{N \times D} $.
