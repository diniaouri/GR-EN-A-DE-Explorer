import dgl.function as fn
import torch
import torch.nn as nn

EOS = 1e-10


class GCNConv_dense(nn.Module):
    """Dense Graph Convolutional Layer.
    
    Performs graph convolution on dense adjacency matrices.
    Implements: H' = A * H * W where A is adjacency, H is features, W is weights.
    
    Args:
        input_size: Dimension of input features
        output_size: Dimension of output features
    """
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCNConv_dgl(nn.Module):
    """DGL-based Graph Convolutional Layer.
    
    Performs graph convolution using DGL's efficient sparse operations.
    Uses message passing with weighted edges for flexible graph structures.
    
    Args:
        input_size: Dimension of input features
        output_size: Dimension of output features
    """
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        """Forward pass using DGL message passing.
        
        Args:
            x: Input node features
            g: DGL graph with edge weights in g.edata['w']
            
        Returns:
            Aggregated node features
        """
        with g.local_scope():
            # Apply linear transformation to node features
            g.ndata['h'] = self.linear(x)
            # Message passing: multiply node features by edge weights and sum
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class Attentive(nn.Module):
    """Diagonal attention layer.
    
    Applies learnable element-wise scaling to input features.
    Equivalent to attention with diagonal weight matrix.
    
    Args:
        isize: Size of input features
    """
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        """Apply diagonal attention weights."""
        return x @ torch.diag(self.w)


class SparseDropout(nn.Module):
    """Dropout layer for sparse tensors.
    
    Applies dropout to sparse tensor values while maintaining sparsity pattern.
    Unlike standard dropout, this preserves the sparse structure of the input.
    
    Args:
        dprob: Dropout probability (fraction of values to drop)
    """
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        """Apply dropout to sparse tensor values.
        
        Args:
            x: Sparse tensor to apply dropout to
            
        Returns:
            Sparse tensor with dropout applied and rescaled values
            
        Note:
            Values are rescaled by 1/kprob to maintain expected sum during training.
        """
        # Create binary mask: randomly keep values with probability kprob
        # torch.rand generates uniform [0,1), adding kprob shifts to [kprob, 1+kprob)
        # floor() converts values >= 1 to True (kept), < 1 to False (dropped)
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        # Extract indices and values for kept elements
        rc = x._indices()[:,mask]
        # Rescale values by 1/kprob to compensate for dropped elements
        # This maintains the expected value during training
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)