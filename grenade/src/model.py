import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU

class GCN(nn.Module):
    """Graph Convolutional Network for node classification and representation learning.
    
    Implements a multi-layer GCN with dropout and adjustable sparsity support.
    Can operate on either dense or sparse adjacency matrices.
    
    Args:
        in_channels: Number of input features
        hidden_channels: Number of hidden layer features
        out_channels: Number of output features
        num_layers: Total number of GCN layers
        dropout: Dropout rate for node features
        dropout_adj: Dropout rate for adjacency matrix
        Adj: Input adjacency matrix (dense or DGL graph)
        sparse: If True, uses sparse operations
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse
        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):
        """Forward pass through the GCN.
        
        Args:
            x: Input node features tensor
            
        Returns:
            Output node features after GCN layers
        """
        # Apply dropout to adjacency matrix
        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)
        # Apply GCN layers with ReLU and dropout (except for last layer)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer without activation
        x = self.layers[-1](x, Adj)
        return x

class GraphEncoder(nn.Module):
    """Graph encoder for contrastive learning.
    
    Encodes graph-structured data using GCN layers and a projection head.
    Used in contrastive learning frameworks to generate embeddings.
    
    Args:
        nlayers: Number of GCN layers
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        emb_dim: Embedding dimension
        proj_dim: Projection head output dimension
        dropout: Dropout rate for node features
        dropout_adj: Dropout rate for adjacency matrix
        sparse: If True, uses sparse graph operations
    """
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse
        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))
        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_, branch=None):
        """Forward pass through the graph encoder.
        
        ========================================================================
        UNDERSTANDING THE ANCHOR VIEW CONCEPT
        ========================================================================
        
        In contrastive learning, we need TWO views of the same data:
        
        1. **ANCHOR VIEW** (branch='anchor'):
           - The "original" or "reference" view of the graph
           - Uses a deep copy of the adjacency matrix to preserve it
           - This view is kept consistent across the training step
           - Think of it as the "ground truth" representation
        
        2. **AUGMENTED VIEW** (branch=None or anything else):
           - A "perturbed" or "alternative" view of the same graph
           - May have dropout applied, edges added/removed, etc.
           - This tests if the model can recognize the same posts despite changes
           - Shares the adjacency matrix reference (no deep copy needed)
        
        **Why the deep copy for anchor?**
        When we apply dropout to the adjacency matrix (Adj.edata['w']), it
        modifies the tensor in-place. The anchor view needs its own copy so
        that dropout applied to the augmented view doesn't affect it.
        
        **The contrastive learning flow:**
        ```
        1. anchor_emb, _ = encoder(features, anchor_graph, branch='anchor')
        2. aug_emb, _ = encoder(features, aug_graph, branch='augmented')  
        3. loss = contrastive_loss(anchor_emb, aug_emb)
        ```
        
        The model learns to produce similar embeddings for the same post
        regardless of which graph view (anchor or augmented) is used.
        
        ========================================================================
        
        Args:
            x: Input node features tensor of shape (num_nodes, feature_dim)
            Adj_: Adjacency matrix (dense tensor or DGL graph with edge weights)
            branch: Optional identifier for the view type:
                   - 'anchor': Creates deep copy to preserve original graph
                   - None or other: Uses shared reference for augmented view
            
        Returns:
            Tuple of:
            - z: Projected embeddings for contrastive learning (num_nodes, proj_dim)
            - x: Raw embeddings before projection (num_nodes, emb_dim)
            
        Note:
            The projection head (proj_head) maps embeddings to a space optimized
            for contrastive learning, while raw embeddings can be used for
            downstream tasks like classification.
        """
        # Apply dropout to adjacency matrix
        # For anchor: deep copy preserves original, for augmented: modify in-place
        if self.sparse:
            if branch == 'anchor':
                # ANCHOR VIEW: Deep copy to preserve original adjacency
                Adj = copy.deepcopy(Adj_)
            else:
                # AUGMENTED VIEW: Use reference (will be modified by dropout)
                Adj = Adj_
            # Apply edge dropout for graph augmentation
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            # Dense adjacency: dropout module handles the augmentation
            Adj = self.dropout_adj(Adj_)
            
        # Apply GCN layers with activation and dropout
        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final GCN layer without activation (raw embeddings)
        x = self.gnn_encoder_layers[-1](x, Adj)
        
        # Project embeddings to contrastive learning space
        # This projection head helps create better representations for contrastive learning
        z = self.proj_head(x)
        
        return z, x

def contrastive_loss(x, x_aug, temperature=0.2, sym=True):
    """Compute contrastive loss between original and augmented embeddings.
    
    ============================================================================
    HOW THE LOSS WORKS (Step-by-Step)
    ============================================================================
    
    1. **Compute Similarity Matrix**
       - Calculate cosine similarity between ALL pairs: x[i] and x_aug[j]
       - Results in a (batch_size x batch_size) matrix
       - Diagonal = similarity of matching pairs (positive)
       - Off-diagonal = similarity of non-matching pairs (negative)
    
    2. **Apply Temperature Scaling**
       - Divide similarities by temperature (default 0.2)
       - Lower temperature → model becomes more discriminative
       - Higher temperature → softer distinctions
       - Then exponentiate to get positive values for probability
    
    3. **Compute Loss**
       - For each post i, we want: exp(sim(x[i], x_aug[i])) / sum_j(exp(sim(x[i], x_aug[j])))
       - This is like asking: "What fraction of total similarity goes to the correct match?"
       - We want this fraction to be HIGH (close to 1.0)
       - Loss = -log(this fraction) → minimizing loss maximizes the fraction
    
    4. **Symmetric Mode** (if sym=True)
       - Compute loss in both directions: x→x_aug AND x_aug→x
       - Average them for more stable training
       - Recommended for most cases
    
    ============================================================================
    MATHEMATICAL DETAILS
    ============================================================================
    
    This implements NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    used in SimCLR and other contrastive learning frameworks.
    
    Formula (for one direction):
        loss_i = -log( exp(sim(z_i, z̃_i) / τ) / Σ_j exp(sim(z_i, z̃_j) / τ) )
    
    where:
        - z_i is embedding of sample i in view 1 (anchor) → corresponds to x[i]
        - z̃_i is embedding of sample i in view 2 (augmented) → corresponds to x_aug[i]
        - τ is temperature parameter
        - j ranges over all samples in the batch
        
    Note: In the code, we use 'x' for original/anchor embeddings and 'x_aug' 
    for augmented embeddings. The tilde notation (z̃) in the mathematical formula 
    represents the augmented embeddings (x_aug). The terms "original" and "anchor" 
    are used interchangeably to refer to the base view before augmentation.
    
    ============================================================================
    
    Args:
        x: Original embeddings tensor of shape (batch_size, embedding_dim)
           These come from the ANCHOR view (base graph structure)
        x_aug: Augmented embeddings tensor of shape (batch_size, embedding_dim)
               These come from the AUGMENTED view (perturbed graph structure)
        temperature: Temperature scaling parameter (default: 0.2)
                    Lower values make the model more discriminative
                    Typical range: 0.1 to 0.5
        sym: If True, computes symmetric loss (average of both directions)
             If False, only computes loss in one direction
             Symmetric mode is more stable and recommended
    
    Returns:
        Scalar contrastive loss value (lower is better)
        The loss encourages:
        - HIGH similarity between positive pairs (same post, different views)
        - LOW similarity between negative pairs (different posts)
        
    Example:
        >>> # After getting embeddings from anchor and augmented graphs
        >>> anchor_embeddings = model(features, anchor_graph)
        >>> aug_embeddings = model(features, augmented_graph)
        >>> loss = contrastive_loss(anchor_embeddings, aug_embeddings)
        >>> # This loss trains the model to recognize posts across different graph views
    """
    batch_size, _ = x.size()
    
    # Step 1: Compute L2 norms for normalization (for cosine similarity)
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    
    # Step 2: Compute similarity matrix using Einstein summation notation
    # sim_matrix[i,j] = cosine_similarity(x[i], x_aug[j])
    # einsum('ik,jk->ij') computes dot products: x[i] · x_aug[j] for all i,j
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    
    # Step 3: Apply temperature scaling and exponentiate
    # This converts similarities to positive values suitable for softmax
    sim_matrix = torch.exp(sim_matrix / temperature)
    
    # Step 4: Extract diagonal (positive pairs - same post in both views)
    # pos_sim[i] = similarity between x[i] and x_aug[i]
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    
    if sym:
        # SYMMETRIC LOSS: Compute in both directions and average
        
        # Direction 1: How well does x[i] match x_aug[i] among all x_aug[j]?
        # Denominator excludes the positive pair itself (- pos_sim)
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        
        # Direction 2: How well does x_aug[i] match x[i] among all x[j]?
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        
        # Take negative log (we want to maximize the ratio, minimize the loss)
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        
        # Average both directions for stability
        contrastive_loss_value = (loss_0 + loss_1) / 2.0
    else:
        # ASYMMETRIC LOSS: Compute only in one direction
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        contrastive_loss_value = - torch.log(loss_1).mean()
    
    return contrastive_loss_value

class GCL(nn.Module):
    """Graph Contrastive Learning model wrapper.
    
    Wraps a GraphEncoder for contrastive learning on graphs. This is the main
    model interface for training with contrastive objectives.
    
    Args:
        nlayers: Number of GCN layers
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        emb_dim: Embedding dimension
        proj_dim: Projection head output dimension
        dropout: Dropout rate for node features
        dropout_adj: Dropout rate for adjacency matrix
        sparse: If True, uses sparse graph operations
    """
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding
