import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy import sparse, stats
import pickle
import torch.nn.functional as F
from    plt.figure(figsize=FIGURE_SETTINGS['figsize'], dpi=FIGURE_SETTINGS['dpi'])
    
    # Calculate correlation coefficients
    corr_ntky_mem = df['align_ntk_y'].corr(df['percent_memorized'])
    corr_ntka_mem = df['align_ntk_a'].corr(df['percent_memorized'])
    
    # Define colors for each alignment type
    ntky_color = 'royalblue'  # Color for kernel-target alignment
    ntka_color = 'purple'     # Color for kernel-graph alignment
    
    # Set proper axis limits based on the data with more padding on the bottom
    min_align = min(df['align_ntk_a'].min(), df['align_ntk_y'].min())
    max_align = max(df['align_ntk_a'].max(), df['align_ntk_y'].max())
    x_padding = (max_align - min_align) * 0.1  # 10% padding
    plt.xlim(min_align - x_padding, max_align + x_padding)
    plt.ylim(0, 80)       # Slightly higher than max memorization rateimport empirical_ntk, centered_kernel_alignment
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, to_undirected
from scipy.sparse import csr_matrix

from dataset import CustomDataset
from model import NodeGCN, NodeGAT, NodeGraphConv
from memorization import calculate_node_memorization_score
from main import (set_seed, train_models, verify_no_data_leakage, 
                 setup_logging, get_model, test)
from nodeli import li_node
from neuron_analysis import (analyze_neuron_flipping, analyze_neuron_flipping_with_memorization,
                           plot_neuron_flipping_analysis, plot_neuron_flipping_by_memorization)


def construct_sparse_adj(edge_index, num_nodes, type='DAD', device='cpu'):
    """Constructs a sparse normalized adjacency matrix (D^-0.5 * (A+I) * D^-0.5)."""
    # Ensure edge_index is undirected for symmetric normalization
    undirected_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    sparse_adj = to_scipy_sparse_matrix(undirected_edge_index, num_nodes=num_nodes)
    adj_loop = sparse_adj + sparse.eye(num_nodes, format='csr')
    rowsum = np.array(adj_loop.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt, format='csr')
    normalized_adj = adj_loop.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()
    
    # Convert to PyTorch sparse CSR tensor
    adj_torch = torch.sparse_csr_tensor(torch.from_numpy(normalized_adj.indptr).long(),
                                        torch.from_numpy(normalized_adj.indices).long(),
                                        torch.from_numpy(normalized_adj.data),
                                        size=(num_nodes, num_nodes),
                                        dtype=torch.float32).to(device)
    return adj_torch

def plot_correlation(df, x_col, y_col, save_path, title, plot_type=None, ax_index=None):
    """Creates a scatter plot for correlation analysis with custom styling."""
    if plot_type == 'composite':
        # This will be handled by create_composite_ntk_plot
        return
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    # Add regression line
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='red')
    
    # Set title and labels with LaTeX formatting
    if x_col == 'align_ntk_a' and y_col == 'percent_memorized':
        # Plot 3: NTK-Graph Alignment vs Memorization Rate
        plt.xlabel(r'$A(\Theta_{final}, A)$', fontsize=20)  # Final NTK-Graph Alignment
        plt.ylabel(r'$MR (\%)$', fontsize=20)               # Memorization Rate
    elif x_col == 'align_ntk_y' and y_col == 'percent_memorized':
        # Plot 4: NTK-Label Alignment vs Memorization Rate
        plt.xlabel(r'$A(\Theta_{final}, \Theta^*)$', fontsize=20)  # Final NTK-Label Alignment
        plt.ylabel(r'$MR (\%)$', fontsize=20)                      # Memorization Rate
    elif x_col == 'align_a_y' and y_col == 'percent_memorized':
        # Plot 1: Graph-Label Alignment vs Memorization Rate
        plt.xlabel(r'$A(A, \Theta^*)$', fontsize=20)  # Graph-Label Alignment
        plt.ylabel(r'$MR (\%)$', fontsize=20)         # Memorization Rate
    elif x_col == 'align_a_y' and y_col == 'align_ntk_y':
        # Plot 2: Graph-Label Alignment vs Final NTK-Label Alignment
        plt.xlabel(r'$A(A, \Theta^*)$', fontsize=20)          # Graph-Label Alignment
        plt.ylabel(r'$A(\Theta_{final}, \Theta^*)$', fontsize=20)  # Final NTK-Label Alignment
    else:
        # Fall back to default labels for any other combinations
        plt.xlabel(x_col.replace('_', ' ').title(), fontsize=14)
        if y_col == 'percent_memorized':
            plt.ylabel(r'$MR (\%)$', fontsize=20)
        else:
            plt.ylabel(y_col.replace('_', ' ').title(), fontsize=14)
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved correlation plot: {save_path}")

def load_and_process_dataset(args, dataset_name, logger):
    """Load synthetic Cora dataset and convert to PyG format"""
    # Construct full path to dataset
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syn-cora")
    dataset = CustomDataset(root=root_dir, name=dataset_name, setting="gcn")
    
    # Convert to PyG format
    edge_index = torch.from_numpy(np.vstack(dataset.adj.nonzero())).long()
    
    # Convert sparse features to dense numpy array
    if sparse.issparse(dataset.features):
        x = torch.from_numpy(dataset.features.todense()).float()
    else:
        x = torch.from_numpy(dataset.features).float()
    
    y = torch.from_numpy(dataset.labels).long()
    
    # Create train/val/test masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    
    train_mask[dataset.idx_train] = True
    val_mask[dataset.idx_val] = True
    test_mask[dataset.idx_test] = True
    
    # Convert to networkx for informativeness calculation
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(edge_index.t().numpy())
    
    # Calculate label informativeness using existing function
    informativeness = li_node(G, dataset.labels)
    
    # Calculate homophily (edge homophily)
    edges = edge_index.t().numpy()
    same_label = dataset.labels[edges[:, 0]] == dataset.labels[edges[:, 1]]
    homophily = same_label.mean()
    
    # Create a data object
    data = type('Data', (), {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': len(y),
        'informativeness': informativeness,
        'homophily': homophily
    })()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Number of features: {x.shape[1]}")
    logger.info(f"Number of classes: {len(torch.unique(y))}")
    logger.info(f"Homophily: {homophily:.4f}")
    logger.info(f"Label Informativeness: {informativeness:.4f}")
    
    return data

def create_visualization(results_df, save_path, args):
    """Create scatter plot of homophily vs informativeness colored by memorization rate"""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    plt.xlabel('Homophily', fontsize=20,font='Sans Serif')
    plt.ylabel('Label Informativeness', fontsize=20,font='Sans Serif')
    #plt.title(f'Memorization Analysis\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits using all available training nodes.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    num_train_nodes = len(train_indices)
    
    # Calculate split sizes: 50% shared, 25% candidate, 25% independent
    shared_size = int(0.50 * num_train_nodes)
    remaining = num_train_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    independent_idx = train_indices[shared_size + split_size:].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, independent_idx, candidate_idx
    else:
        return shared_idx, candidate_idx, independent_idx

def analyze_node_dropping(model_f, model_g, data, nodes_dict, node_scores, device, drop_percentage=0.05, logger=None):
    """
    Analyze the impact of dropping nodes with high/low memorization scores on model performance.
    
    Args:
        model_f: First GNN model
        model_g: Second GNN model
        data: PyG data object
        nodes_dict: Dictionary containing node indices for each category
        node_scores: Dictionary containing memorization scores
        device: torch device
        drop_percentage: Target percentage of nodes to drop (will adjust to maintain equal numbers)
        logger: Logger object
    """
    results = {}
    
    # Get memorization scores for shared and candidate nodes
    shared_scores = pd.DataFrame({
        'node_idx': nodes_dict['shared'],
        'mem_score': node_scores['shared']['mem_scores']
    })
    candidate_scores = pd.DataFrame({
        'node_idx': nodes_dict['candidate'],
        'mem_score': node_scores['candidate']['mem_scores']
    })
    
    # Calculate number of nodes to drop (use min to ensure equal numbers)
    shared_low_mem = shared_scores[shared_scores['mem_score'] < 0.5]
    candidate_high_mem = candidate_scores[candidate_scores['mem_score'] > 0.5]
    n_nodes_to_drop = min(
        len(shared_low_mem),
        len(candidate_high_mem),
        max(1, int(min(len(shared_scores), len(candidate_scores)) * drop_percentage))
    )
    
    actual_drop_percentage = n_nodes_to_drop / min(len(shared_scores), len(candidate_scores))
    
    if logger:
        logger.info(f"\nNode Dropping Analysis:")
        logger.info(f"Target drop percentage: {drop_percentage:.1%}")
        logger.info(f"Actual drop percentage: {actual_drop_percentage:.1%}")
        logger.info(f"Number of nodes to drop from each category: {n_nodes_to_drop}")
    
    # Get nodes to drop
    shared_to_drop = (shared_low_mem
                     .sort_values('mem_score')
                     .head(n_nodes_to_drop)['node_idx'].tolist())
    
    candidate_to_drop = (candidate_high_mem
                        .sort_values('mem_score', ascending=False)
                        .head(n_nodes_to_drop)['node_idx'].tolist())
    
    # Function to create new training mask excluding specified nodes
    def create_modified_mask(nodes_to_exclude):
        mask = data.train_mask.clone()
        mask[nodes_to_exclude] = False
        return mask
    
    # Baseline performance (no nodes dropped)
    baseline_acc = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
    
    # Test performance after dropping non-memorized shared nodes
    train_mask_no_low_mem = create_modified_mask(shared_to_drop)
    acc_no_low_mem = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
    
    # Test performance after dropping memorized candidate nodes
    train_mask_no_high_mem = create_modified_mask(candidate_to_drop)
    acc_no_high_mem = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
    
    results = {
        'baseline_acc': baseline_acc,
        'acc_without_low_mem': acc_no_low_mem,
        'acc_without_high_mem': acc_no_high_mem,
        'n_nodes_dropped': n_nodes_to_drop,
        'actual_drop_percentage': actual_drop_percentage,
        'shared_nodes_dropped': shared_to_drop,
        'candidate_nodes_dropped': candidate_to_drop
    }
    
    if logger:
        logger.info("\nTest Accuracies:")
        logger.info(f"Baseline: {baseline_acc:.4f}")
        logger.info(f"Without low-mem shared nodes: {acc_no_low_mem:.4f}")
        logger.info(f"Without high-mem candidate nodes: {acc_no_high_mem:.4f}")
    
    return results

def plot_node_dropping_vs_homophily(results_df, save_path, logger=None):
    """
    Create a plot showing how dropping nodes affects performance across homophily levels.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot lines with confidence intervals
    sns.lineplot(data=results_df, x='homophily', y='baseline_acc', 
                label='Baseline', marker='o')
    sns.lineplot(data=results_df, x='homophily', y='acc_without_low_mem',
                label='Without Low-Mem Shared', marker='s')
    sns.lineplot(data=results_df, x='homophily', y='acc_without_high_mem',
                label='Without High-Mem Candidate', marker='^')
    
    plt.xlabel('Homophily Level')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Node Dropping Across Homophily Levels')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text box with statistics
    stats_text = []
    for h_level in results_df['homophily'].unique():
        subset = results_df[results_df['homophily'] == h_level]
        # Perform t-test between baseline and dropping high-mem nodes
        t_stat, p_val = stats.ttest_rel(
            subset['baseline_acc'],
            subset['acc_without_high_mem']
        )
        stats_text.append(f'h={h_level:.1f}: t={t_stat:.2f}, p={p_val:.3f}')
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("\nStatistical Analysis of Node Dropping Impact:")
        logger.info("\n".join(stats_text))

def plot_combined_ntk_vs_memorization(df, save_path):
    """Creates a combined scatter plot for NTK alignments vs Memorization Rate."""
    # Global plot parameters matching finalntkplots_new.py
    FIGURE_SETTINGS = {
        'figsize': (25, 20),
        'dpi': 500
    }
    
    LINE_SETTINGS = {
        'linewidth': 15,
        'markersize': 30,
        'alpha': 0.8
    }
    
    FONT_SIZES = {
        'label': 60,
        'tick': 60,
        'legend': 40,
        'title': 40
    }
    # Global plot parameters matching finalntkplots_new.py
    FIGURE_SETTINGS = {
        'figsize': (25, 20),
        'dpi': 500
    }
    
    LINE_SETTINGS = {
        'linewidth': 15,
        'markersize': 30,
        'alpha': 0.8
    }
    
    FONT_SIZES = {
        'label': 60,
        'tick': 60,
        'legend': 40,
        'title': 40
    }
    
    plt.figure(figsize=FIGURE_SETTINGS['figsize'], dpi=FIGURE_SETTINGS['dpi'])
    
    # Calculate correlation coefficients
    corr_ntky_mem = df['align_ntk_y'].corr(df['percent_memorized'])
    corr_ntka_mem = df['align_ntk_a'].corr(df['percent_memorized'])
    
    # Define colors for each alignment type
    ntky_color = 'royalblue'  # Color for kernel-target alignment
    ntka_color = 'purple'     # Color for kernel-graph alignment
    
    # Set proper axis limits based on the data with more padding on the bottom
    min_align = min(df['align_ntk_a'].min(), df['align_ntk_y'].min())
    max_align = max(df['align_ntk_a'].max(), df['align_ntk_y'].max())
    x_padding = (max_align - min_align) * 0.1  # 10% padding
    plt.xlim(min_align - x_padding, max_align + x_padding)
    plt.ylim(0, 80)       # Slightly higher than max memorization rate
    
    # Sort dataframe by x values to ensure smooth trend lines
    df_sorted_y = df.sort_values('align_ntk_y')
    df_sorted_a = df.sort_values('align_ntk_a')
    
    # Plot Kernel-Target Alignment vs Memorization Rate using numpy polyfit for better control
    coeffs_y = np.polyfit(df['align_ntk_y'], df['percent_memorized'], 1)
    poly_y = np.poly1d(coeffs_y)
    
    # Scatter plot for NTK-Y
    plt.scatter(df['align_ntk_y'], df['percent_memorized'], 
               alpha=LINE_SETTINGS['alpha'], s=LINE_SETTINGS['markersize']*20,
               color=ntky_color, label=fr'$A(\Theta_{{final}}, \Theta^*)$ vs MR (r={corr_ntky_mem:.2f})')
    
    # Add trend line for NTK-Y
    x_range_y = np.linspace(df['align_ntk_y'].min(), df['align_ntk_y'].max(), 100)
    plt.plot(x_range_y, poly_y(x_range_y), '-', color=ntky_color, 
            linewidth=LINE_SETTINGS['linewidth'])
    
    # Plot Kernel-Graph Alignment vs Memorization Rate using numpy polyfit
    coeffs_a = np.polyfit(df['align_ntk_a'], df['percent_memorized'], 1)
    poly_a = np.poly1d(coeffs_a)
    
    # Scatter plot for NTK-A
    plt.scatter(df['align_ntk_a'], df['percent_memorized'],
               alpha=LINE_SETTINGS['alpha'], s=LINE_SETTINGS['markersize']*20,
               color=ntka_color, label=fr'$A(\Theta_{{final}}, A)$ vs MR (r={corr_ntka_mem:.2f})')
    
    # Add trend line for NTK-A
    x_range_a = np.linspace(df['align_ntk_a'].min(), df['align_ntk_a'].max(), 100)
    plt.plot(x_range_a, poly_a(x_range_a), '--', color=ntka_color,
            linewidth=LINE_SETTINGS['linewidth'])

    plt.xlabel('NTK Alignment Value', fontsize=FONT_SIZES['label'], font='Sans Serif')
    plt.ylabel(r'Memorization Rate (MR %)', fontsize=FONT_SIZES['label'], font='Sans Serif')
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    plt.legend(fontsize=FONT_SIZES['legend'], loc='upper right', framealpha=0.8,
             edgecolor='gray', borderaxespad=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_SETTINGS['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved combined NTK correlation plot: {save_path}")

def create_composite_ntk_plot(results_df, save_path, timestamp):
    """
    Creates a composite figure with multiple subplots showing NTK alignments
    with custom styling as specified in the requirements.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'hspace': 0.3})
    
    # Subplot 0: Test Accuracy vs Homophily
    sns.lineplot(data=results_df, x='homophily', y='baseline_acc', ax=axes[0], marker='o')
    axes[0].set_title('(a) Synthetic Dataset')
    axes[0].set_ylabel('Test Acc.', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 1: NTK-Graph Alignment (Theta-A) vs Homophily
    sns.lineplot(data=results_df, x='homophily', y='align_ntk_a', ax=axes[1], marker='s', color='purple')
    axes[1].set_ylabel('$A$($\\Theta_t$,A)', color='purple', fontsize=12)
    axes[1].tick_params(axis='y', labelcolor='purple')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: Graph-Label Alignment (A-Theta*) vs Homophily
    sns.lineplot(data=results_df, x='homophily', y='align_a_y', ax=axes[2], marker='^', color='firebrick')
    axes[2].set_ylabel('$A$(A,$\\Theta^*$)', color='firebrick', fontsize=12)
    axes[2].tick_params(axis='y', labelcolor='firebrick')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 3: NTK-Label Alignment (Theta-Theta*) vs Homophily
    sns.lineplot(data=results_df, x='homophily', y='align_ntk_y', ax=axes[3], marker='d', color='royalblue')
    axes[3].set_ylabel('$A$($\\Theta_t$,$\\Theta^*$)', color='royalblue', fontsize=12)
    axes[3].set_xlabel('Homophily', fontsize=14)
    axes[3].tick_params(axis='y', labelcolor='royalblue')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved composite NTK plot: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gcn')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/syncora_ntk_analysis')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_ntk_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with both file and console output
    logger = logging.getLogger('syncora_ntk_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Select homophily levels to analyze
    homophily_levels = [0.0, 0.3, 0.5, 0.7, 1.0]  # Changed to match available files
    dataset_files = [f'h{h:.2f}-r1' for h in homophily_levels]
    
    # Initialize results containers
    results = []  # Main results
    dropping_results = []  # Node dropping analysis results
    neuron_results = {}  # Added initialization for neuron analysis results
    
    for dataset_name in tqdm(dataset_files, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load and process dataset
        data = load_and_process_dataset(args, dataset_name, logger)
        
        # Move individual tensors to device instead of entire Data object
        data.x = data.x.to(device) if hasattr(data, 'x') else None
        data.edge_index = data.edge_index.to(device) if hasattr(data, 'edge_index') else None
        data.y = data.y.to(device) if hasattr(data, 'y') else None
        if hasattr(data, 'train_mask'):
            data.train_mask = data.train_mask.to(device)
        if hasattr(data, 'val_mask'):
            data.val_mask = data.val_mask.to(device)
        if hasattr(data, 'test_mask'):
            data.test_mask = data.test_mask.to(device)
        
        # Compute Theta_star (optimal kernel)
        num_classes = len(torch.unique(data.y))  # Get num_classes dynamically
        Y_onehot = F.one_hot(data.y, num_classes=num_classes).float().to(device)  # Ensure on device
        Theta_star = Y_onehot @ Y_onehot.t()
        logger.info("Computed Theta* (Optimal Kernel)")
        
        # Compute A_dense (effective graph structure)
        logger.info("Computing A_dense (effective graph structure)...")
        A_sparse = construct_sparse_adj(data.edge_index,
                                        num_nodes=data.num_nodes,
                                        type='DAD',
                                        device=device)
        
        A_dense = torch.eye(data.num_nodes, device=device)
        # Propagate A sparse num_layers * 2 times
        num_propagations = args.num_layers * 2
        logger.info(f"Computing A_dense with {num_propagations} propagations...")
        with torch.no_grad():
            for _ in range(num_propagations):
                # Use sparse matrix multiplication
                A_dense = torch.sparse.mm(A_sparse, A_dense)
        
        # Convert final A_dense to dense if it's sparse
        if A_dense.is_sparse:
            A_dense = A_dense.to_dense()
        logger.info("Computed A_dense (Effective Graph Structure)")
        
        # Get node splits
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=False
        )
        
        # Get extra indices from test set
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Train models
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None
        )
        
        # Calculate memorization scores
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Compute final NTK and alignments
        try:
            logger.info("Computing final empirical NTK for model_f...")
            model_f.to(device)  # Ensure model is on device
            data_x_dev = data.x.to(device)  # Ensure features are on device
            edge_index_dev = data.edge_index.to(device)  # Ensure edge_index is on device
            Theta_final = empirical_ntk(model_f, data_x_dev, edge_index=edge_index_dev, type='nn')  # Compute n x n NTK
            logger.info("Final NTK computed.")
            
            logger.info("Computing final alignments...")
            # Ensure matrices are on the correct device for alignment function
            align_ntk_a = centered_kernel_alignment(Theta_final, A_dense)
            align_ntk_y = centered_kernel_alignment(Theta_final, Theta_star)
            align_a_y = centered_kernel_alignment(A_dense, Theta_star)  # Intrinsic homophily measure
            logger.info(f"Alignments: NTK-A={align_ntk_a:.4f}, NTK-Y={align_ntk_y:.4f}, A-Y={align_a_y:.4f}")
            
        except Exception as e:
            logger.error(f"Error during NTK computation or alignment: {e}", exc_info=True)
            align_ntk_a, align_ntk_y, align_a_y = np.nan, np.nan, np.nan
        
        # Perform node dropping analysis
        dropping_analysis = analyze_node_dropping(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            node_scores=node_scores,
            device=device,
            drop_percentage=0.05,  # 5% default
            logger=logger
        )
        
        # Store dropping results with homophily level
        dropping_results.append({
            'homophily': float(data.homophily),
            'baseline_acc': dropping_analysis['baseline_acc'],
            'acc_without_low_mem': dropping_analysis['acc_without_low_mem'],
            'acc_without_high_mem': dropping_analysis['acc_without_high_mem'],
            'n_nodes_dropped': dropping_analysis['n_nodes_dropped'],
            'actual_drop_percentage': dropping_analysis['actual_drop_percentage']
        })
        
        # Store other results with the new alignment values
        results.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc),
            'baseline_acc': dropping_analysis['baseline_acc'],
            'acc_drop_low_mem': dropping_analysis['acc_without_low_mem'],
            'acc_drop_high_mem': dropping_analysis['acc_without_high_mem'],
            'align_ntk_a': align_ntk_a,
            'align_ntk_y': align_ntk_y,
            'align_a_y': align_a_y
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    dropping_df = pd.DataFrame(dropping_results)
    
    # Create memorization visualization
    plot_path = os.path.join(log_dir, f'memorization_analysis_{timestamp}.png')
    create_visualization(results_df, plot_path, args)
    
    # Create node dropping vs homophily visualization
    dropping_plot_path = os.path.join(log_dir, f'node_dropping_analysis_{timestamp}.png')
    plot_node_dropping_vs_homophily(dropping_df, dropping_plot_path, logger)
    
    # Create NTK alignment correlation plots
    # Plot: Memorization Rate vs. Intrinsic Homophily (A-Y Alignment)
    plot_corr_path_mem_ay = os.path.join(log_dir, f'corr_mem_vs_align_a_y_{timestamp}.png')
    plot_correlation(results_df, 'align_a_y', 'percent_memorized',
                     plot_corr_path_mem_ay, 'Memorization Rate vs. Graph-Label Alignment (Homophily)')
    
    # Plot: Final NTK-Y Alignment vs. Intrinsic Homophily (A-Y Alignment)
    plot_corr_path_ntky_ay = os.path.join(log_dir, f'corr_align_ntky_vs_align_a_y_{timestamp}.png')
    plot_correlation(results_df, 'align_a_y', 'align_ntk_y',
                     plot_corr_path_ntky_ay, 'Final NTK-Label Alignment vs. Graph-Label Alignment (Homophily)')
    
    # Plot: Memorization Rate vs. Final NTK-Y Alignment
    plot_corr_path_mem_ntky = os.path.join(log_dir, f'corr_mem_vs_align_ntky_{timestamp}.png')
    plot_correlation(results_df, 'align_ntk_y', 'percent_memorized',
                     plot_corr_path_mem_ntky, 'Memorization Rate vs. Final NTK-Label Alignment')
    
    # Optional: Plot Memorization Rate vs. Final NTK-A Alignment
    plot_corr_path_mem_ntka = os.path.join(log_dir, f'corr_mem_vs_align_ntka_{timestamp}.png')
    plot_correlation(results_df, 'align_ntk_a', 'percent_memorized',
                     plot_corr_path_mem_ntka, 'Memorization Rate vs. Final NTK-Graph Alignment')
    
    # Create combined NTK vs memorization plot
    combined_ntk_mem_path = os.path.join(log_dir, f'combined_ntk_vs_memorization_{timestamp}.pdf')
    plot_combined_ntk_vs_memorization(results_df, combined_ntk_mem_path)
    
    # Create composite NTK plot
    composite_ntk_path = os.path.join(log_dir, f'composite_ntk_plot_{timestamp}.png')
    create_composite_ntk_plot(results_df, composite_ntk_path, timestamp)
    
    # Save detailed results
    results_df.to_csv(os.path.join(log_dir, 'results_with_ntk.csv'), index=False)
    dropping_df.to_csv(os.path.join(log_dir, 'node_dropping_results.csv'), index=False)
    
    # Log final analysis
    logger.info("\nFinal Analysis Across Homophily Levels:")
    for h in sorted(dropping_df['homophily'].unique()):
        subset = dropping_df[dropping_df['homophily'] == h]
        logger.info(f"\nHomophily Level: {h:.2f}")
        logger.info(f"Baseline Accuracy: {subset['baseline_acc'].mean():.4f} ± {subset['baseline_acc'].std():.4f}")
        logger.info(f"Accuracy without low-mem nodes: {subset['acc_without_low_mem'].mean():.4f} ± {subset['acc_without_low_mem'].std():.4f}")
        logger.info(f"Accuracy without high-mem nodes: {subset['acc_without_high_mem'].mean():.4f} ± {subset['acc_without_high_mem'].std():.4f}")
        
        # Calculate and log confidence intervals (95%)
        for metric in ['baseline_acc', 'acc_without_low_mem', 'acc_without_high_mem']:
            ci = stats.t.interval(0.95, len(subset[metric])-1, 
                                loc=subset[metric].mean(), 
                                scale=stats.sem(subset[metric]))
            logger.info(f"{metric} 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Log NTK alignment analysis
    logger.info("\nNTK Alignment Analysis:")
    # Calculate correlations between metrics
    corr_mem_ntky = results_df['percent_memorized'].corr(results_df['align_ntk_y'])
    corr_mem_ntka = results_df['percent_memorized'].corr(results_df['align_ntk_a'])
    corr_ay_ntky = results_df['align_a_y'].corr(results_df['align_ntk_y'])
    corr_homo_ntky = results_df['homophily'].corr(results_df['align_ntk_y'])
    
    logger.info(f"Correlation between Memorization Rate and NTK-Label Alignment: {corr_mem_ntky:.4f}")
    logger.info(f"Correlation between Memorization Rate and NTK-Graph Alignment: {corr_mem_ntka:.4f}")
    logger.info(f"Correlation between Graph-Label Alignment and NTK-Label Alignment: {corr_ay_ntky:.4f}")
    logger.info(f"Correlation between Homophily and NTK-Label Alignment: {corr_homo_ntky:.4f}")
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Memorization visualization saved as: {plot_path}")
    logger.info(f"Node dropping analysis saved as: {dropping_plot_path}")
    logger.info(f"Combined NTK vs Memorization plot saved as: {combined_ntk_mem_path}")
    logger.info(f"Correlation plots saved to: {log_dir}")

if __name__ == '__main__':
    main()
