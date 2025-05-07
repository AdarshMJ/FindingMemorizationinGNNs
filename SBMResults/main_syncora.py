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
from dataset import CustomDataset
from model import NodeGCN, NodeGAT, NodeGraphConv
from memorization import calculate_node_memorization_score
from main import (set_seed, train_models, verify_no_data_leakage, 
                 setup_logging, get_model, test)
from nodeli import li_node
from neuron_analysis import (analyze_neuron_flipping, analyze_neuron_flipping_with_memorization,
                           plot_neuron_flipping_analysis, plot_neuron_flipping_by_memorization)


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
        s=500,
        alpha=0.7
    )
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Homophily', fontsize=25,font='Sans Serif')
    plt.ylabel('Label Informativeness', fontsize=25,font='Sans Serif')
    #plt.title(f'Memorization Analysis\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=25)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

def create_homophily_vs_memorization_plot(results_df, save_path, args):
    """Create scatter plot of homophily vs memorization rate with trend line"""
    plt.figure(figsize=(10, 8))
    
    # Plot scatter points
    plt.scatter(
        results_df['homophily'],
        results_df['percent_memorized'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=500,
        alpha=0.7
    )
    
    # Add trend line
    z = np.polyfit(results_df['homophily'], results_df['percent_memorized'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results_df['homophily'].min(), results_df['homophily'].max(), 100)
    plt.plot(x_trend, p(x_trend), 'r-', linewidth=3)
    
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('Homophily', fontsize=25, font='Sans Serif')
    plt.ylabel('Memorization Rate (%)', fontsize=25, font='Sans Serif')
    #plt.title(f'Homophily vs. Memorization\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
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
        'n_nodes_dropped': n_nodes_to_drop,  # Fixed from n_nodes_drop to n_nodes_to_drop
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

def visualize_graph(data, save_path, title):
    """
    Visualize graph with nodes colored by labels.
    
    Args:
        data: PyG data object containing graph structure and node labels
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Create a networkx graph from edge_index
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    # Add edges from edge_index
    edges = data.edge_index.cpu().numpy().T
    G.add_edges_from(edges)
    
    # Get node labels and convert to integers for coloring
    labels = data.y.cpu().numpy()
    num_classes = len(np.unique(labels))
    
    # Choose a layout that shows community structure
    if data.homophily < 0.5:
        # For low homophily, use spring layout with stronger repulsion
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    else:
        # For high homophily, use community-aware layout
        pos = nx.kamada_kawai_layout(G)
    
    plt.figure(figsize=(12, 12), dpi=300)
    
    # Generate a color palette based on number of classes
    cmap = plt.cm.get_cmap('tab10', num_classes)
    
    # Draw nodes colored by class
    for label_id in range(num_classes):
        node_indices = np.where(labels == label_id)[0]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_indices,
            node_color=[cmap(label_id)],
            node_size=80,
            alpha=0.9
        )
    
    # Draw edges with transparency
    nx.draw_networkx_edges(
        G, pos,
        width=0.5,
        alpha=0.3,
        edge_color='gray'
    )
    
    # Remove axis and set title
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

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
    parser.add_argument('--output_dir', type=str, default='results/syncora_analysis')
    #parser.add_argument('--analyze_superposition', action='store_true',
     #                  help='Perform superposition analysis to study channel sharing')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with both file and console output
    logger = logging.getLogger('syncora_analysis')
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
        
        # Visualize graphs with homophily values 0.0 and 1.0
        # if dataset_name in ['h0.00-r1', 'h1.00-r1']:
        #     homophily_value = "0.0" if dataset_name == 'h0.00-r1' else "1.0"
        #     logger.info(f"\nVisualizing graph with homophily {homophily_value}...")
        #     graph_path = os.path.join(log_dir, f'graph_visualization_h{homophily_value}.png')
        #     visualize_graph(data, graph_path, f"Graph with Homophily {homophily_value}")
        #     logger.info(f"Graph visualization saved to: {graph_path}")
        
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
        
        # Perform neuron flipping analysis on model_f - original analysis
        # logger.info("\nPerforming neuron flipping analysis...")
        # flip_results = analyze_neuron_flipping(
        #     model_f=model_f,
        #     model_g=model_g,
        #     data=data,
        #     nodes_dict=nodes_dict,
        #     device=device,
        #     logger=logger
        # )
        
        # # Create neuron flipping plot for this homophily level
        # plot_path = os.path.join(log_dir, f'neuron_flipping_{dataset_name}.png')
        # plot_neuron_flipping_analysis(
        #     results=flip_results,
        #     save_path=plot_path,
        #     title=f'Neuron Flipping Analysis (Homophily={data.homophily:.2f})'
        # )
        
        # # NEW: Perform neuron flipping analysis with memorization categorization
        # logger.info("\nPerforming neuron flipping analysis with memorization categorization...")
        # mem_flip_results = analyze_neuron_flipping_with_memorization(
        #     model_f=model_f,
        #     model_g=model_g,
        #     data=data,
        #     nodes_dict=nodes_dict,
        #     memorization_scores=node_scores,
        #     device=device,
        #     threshold=0.5,
        #     logger=logger
        # )
        
        # Create directory for memorization-based plots
        # mem_plot_dir = os.path.join(log_dir, f'mem_neuron_flipping_{dataset_name}')
        # os.makedirs(mem_plot_dir, exist_ok=True)
        
        # # Create plots separated by memorization threshold
        # plot_neuron_flipping_by_memorization(
        #     results=mem_flip_results,
        #     save_dir=mem_plot_dir,
        #     threshold=0.5,
        #     bins=30
        # )
        
        # # Log paths to the new plots
        # logger.info(f"\nMemorization-categorized neuron flipping plots saved to: {mem_plot_dir}")
        
        # # Store neuron flipping results
        # neuron_results[dataset_name] = {
        #     'standard': flip_results,
        #     'by_memorization': mem_flip_results
        # }
        
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
        
        # Store other results as before...
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
            'acc_drop_high_mem': dropping_analysis['acc_without_high_mem']
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    dropping_df = pd.DataFrame(dropping_results)
    
    # Create memorization visualization
    plot_path = os.path.join(log_dir, f'memorization_analysis_{timestamp}.pdf')
    create_visualization(results_df, plot_path, args)
    
    # Create homophily vs memorization plot with trend line
    homophily_plot_path = os.path.join(log_dir, f'homophily_vs_memorization_{timestamp}.pdf')
    create_homophily_vs_memorization_plot(results_df, homophily_plot_path, args)
    
    # Create node dropping vs homophily visualization
    dropping_plot_path = os.path.join(log_dir, f'node_dropping_analysis_{timestamp}.png')
    plot_node_dropping_vs_homophily(dropping_df, dropping_plot_path, logger)
    
    # Save detailed results
    results_df.to_csv(os.path.join(log_dir, 'results.csv'), index=False)
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
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Memorization visualization saved as: {plot_path}")
    logger.info(f"Homophily vs. Memorization plot saved as: {homophily_plot_path}")
    logger.info(f"Node dropping analysis saved as: {dropping_plot_path}")

if __name__ == '__main__':
    main()