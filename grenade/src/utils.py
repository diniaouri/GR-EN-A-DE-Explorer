import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres

EOS = 1e-10

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def edge_deletion(adj, drop_r):
    """Randomly delete edges from adjacency matrix for data augmentation.
    
    Args:
        adj: Adjacency matrix (modified in-place)
        drop_r: Fraction of edges to delete (0 to 1)
        
    Returns:
        Modified adjacency matrix with edges removed
        
    Note:
        Only considers upper triangle to maintain symmetry in undirected graphs.
    """
    edge_index = np.array(np.nonzero(adj))
    # Only consider upper triangle edges to avoid double counting
    half_edge_index = edge_index[:, edge_index[0, :] < edge_index[1, :]]
    num_edge = half_edge_index.shape[1]
    # Randomly sample edges to drop
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    # Remove edges symmetrically
    adj[dropped_edge_index[:, 0], dropped_edge_index[:, 1]] = 0.
    adj[dropped_edge_index[:, 1], dropped_edge_index[:, 0]] = 0.
    return adj

def edge_addition(adj, add_r):
    """Randomly add edges to adjacency matrix for data augmentation.
    
    Args:
        adj: Adjacency matrix (modified in-place)
        add_r: Fraction of edges to add relative to existing edges (0 to 1)
        
    Returns:
        Modified adjacency matrix with edges added
        
    Note:
        New edges are added symmetrically to maintain undirected graph structure.
    """
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0, :] < edge_index[1, :]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    # Randomly sample node pairs for new edges
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    # Add edges symmetrically
    adj[added_edge_index_in, added_edge_index_out] = 1.
    adj[added_edge_index_out, added_edge_index_in] = 1.
    return adj

def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask, samples

def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]

def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj

def nearest_neighbors_sparse(X, k, metric):
    """Compute k-nearest neighbors graph in sparse format.
    
    Creates a sparse adjacency matrix based on k-nearest neighbors with self-loops.
    
    Args:
        X: Feature matrix of shape (num_nodes, feature_dim)
        k: Number of nearest neighbors
        metric: Distance metric (e.g., 'cosine', 'euclidean')
        
    Returns:
        Tuple of (source_indices, target_indices) for sparse edge list
    """
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    # Extract COO format sparse matrix components
    [s_, d_, val] = sp.find(adj)
    # Add self-loops
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d

def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def normalize(adj, mode, sparse=False):
    """Normalize adjacency matrix for graph convolution.
    
    Applies normalization to adjacency matrix to prevent exploding/vanishing
    gradients in GCNs. Supports both dense and sparse formats.
    
    Args:
        adj: Adjacency matrix (dense tensor or sparse tensor)
        mode: Normalization mode
            - "sym": Symmetric normalization D^(-1/2) A D^(-1/2)
            - "row": Row normalization D^(-1) A
        sparse: If True, uses sparse tensor operations
        
    Returns:
        Normalized adjacency matrix in same format as input
        
    Note:
        EOS (epsilon) is added to prevent division by zero for isolated nodes.
    """
    if not sparse:
        if mode == "sym":
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            # Row-wise normalization: D^(-1) A
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        # Sparse tensor operations
        adj = adj.coalesce()
        if mode == "sym":
            # Compute degree^(-1/2) for each node
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            # Apply to both source and target nodes of each edge
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        elif mode == "row":
            # Compute degree^(-1) for each node
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            # Apply only to source nodes
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        # Multiply edge values by normalization factors
        new_values = adj.values() * D_value
        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph

def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.
    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph

def knn_fast(X, k, b):
    """Fast k-nearest neighbors computation with batch processing.
    
    Efficiently computes k-nearest neighbors for large graphs using batch-wise
    similarity computation. Returns edge list format suitable for sparse graphs.
    
    Args:
        X: Node feature matrix of shape (num_nodes, feature_dim)
        k: Number of nearest neighbors to find
        b: Batch size for processing (e.g., 1000)
        
    Returns:
        Tuple of (rows, cols, values) representing edge list:
            - rows: Source node indices
            - cols: Target node indices  
            - values: Normalized edge weights based on similarity
            
    Note:
        Normalizes features and applies symmetric normalization to edge weights
        based on both row and column sums for better graph learning.
    """
    # Normalize features to unit length for cosine similarity
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1))
    rows = torch.zeros(X.shape[0] * (k + 1))
    cols = torch.zeros(X.shape[0] * (k + 1))
    norm_row = torch.zeros(X.shape[0])
    norm_col = torch.zeros(X.shape[0])
    # Process in batches to avoid memory overflow
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        # Compute cosine similarities for current batch
        similarities = torch.mm(sub_tensor, X.t())
        # Find top-k+1 neighbors (including self)
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        # Store results in flat arrays
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        # Accumulate normalization factors
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    # Compute total normalization factor (sum of row and column norms)
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    # Apply symmetric normalization: weight * (norm[source]^-0.5 * norm[target]^-0.5)
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph

def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx

def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)

class clustering_metrics():
    """Metrics for evaluating clustering quality.
    
    Computes various clustering evaluation metrics including accuracy with
    Hungarian algorithm alignment, F1 scores, precision, recall, NMI, and ARI.
    
    Args:
        true_label: Ground truth cluster labels
        predict_label: Predicted cluster labels
    """
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        """Compute clustering accuracy using Hungarian algorithm.
        
        Uses the Munkres (Hungarian) algorithm to find optimal alignment
        between predicted and true clusters, then computes accuracy and
        other metrics based on this alignment.
        
        Returns:
            Tuple of (accuracy, f1_macro, precision_macro, recall_macro,
                     f1_micro, precision_micro, recall_micro)
        """
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0
        # Build cost matrix for Hungarian algorithm
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
                cost[i][j] = len(mps_d)
        # Use Munkres algorithm to find optimal assignment
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)
        # Relabel predictions according to optimal assignment
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        # Compute metrics
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()
        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))
        return acc, nmi, f1_macro, adjscore

class ExperimentParameters:
    """Loads experiment parameters from a CSV configuration file.
    
    Reads a CSV file where each row represents an experiment configuration
    and assigns column values as instance attributes for the specified experiment.
    
    Args:
        exp_nb: Experiment number to load (matches 'exp_nb' column in CSV)
        
    Raises:
        IndexError: If exp_nb not found in the CSV file
        
    Example:
        params = ExperimentParameters(exp_nb=1)
        print(params.learning_rate)  # Access parameter as attribute
    """
    def __init__(self, exp_nb: int) -> None:
        csv_path = "./src/experiment_params.csv"
        # Try utf-8, fallback to utf-16
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", header=0)
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-16", header=0)
        print(f"Loaded CSV shape: {df.shape}")
        print(df.head(10))
        # exp_nb must be matched from the exp_nb column
        if exp_nb not in df['exp_nb'].values:
            raise IndexError(
                f"exp_nb {exp_nb} not found in exp_nb column of the CSV file. Existing exp_nb values: {df['exp_nb'].tolist()}"
            )
        row_idx = df.index[df['exp_nb'] == exp_nb][0]
        row_data = df.loc[row_idx].to_dict()
        for header, value in row_data.items():
            setattr(self, header, value)
    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"{self.__class__.__name__}({attrs})"

def save_loss_plot(loss_values, args):
    plt.figure()
    plt.plot(loss_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution Over Epochs')
    folder_path = 'plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"loss_{current_time}_exp{args.exp_nb}.png"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.close()