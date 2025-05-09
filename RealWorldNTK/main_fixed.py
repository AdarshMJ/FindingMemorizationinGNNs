import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork, WebKB, HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose
import os
import logging
from model import NodeGCN, NodeGAT, NodeGraphConv,NodeGraphSAGE
import numpy as np
import random
from datetime import datetime
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from memorization import calculate_node_memorization_score, plot_node_memorization_analysis
from scipy import stats
from nli_analysis import *
from reliability_analysis import *
from nodeli import get_graph_and_labels_from_pyg_dataset, li_node, h_adj
import csv
# from dimensionality_analysis import plot_memorization_dimensionality
# from analysis import analyze_memorization_vs_misclassification
from dataloader import load_npz_dataset, get_heterophilic_datasets
from knn_label_disagreement import run_knn_label_disagreement_analysis
from knn_disagreement_visualizer import visualize_knn_disagreement_process


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, x, edge_index, train_mask, y, optimizer, device):
    model.train()
    optimizer.zero_grad()
    # Ensure all tensors are on the same device
    x = x.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    y = y.to(device)
    
    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, mask, y, device):
    model.eval()
    with torch.no_grad():
        # Ensure all tensors are on the same device
        x = x.to(device)
        edge_index = edge_index.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        out = model(x, edge_index)
        pred = out[mask].max(1)[1]
        correct = pred.eq(y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with the structure ModelType_Datasetname_NumLayers_timestamp
    dir_name = f"{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}"
    
    # Create base results directory if it doesn't exist
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create full directory path
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, f'{args.model_type}_{args.dataset}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_dir, timestamp

def load_dataset(args):
    transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])
    
    # Handle standard PyTorch Geometric datasets
    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['computers', 'photo']:
        dataset = Amazon(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() == 'actor':
        dataset = Actor(root='data/Actor', transform=transforms)
    elif args.dataset.lower() in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['cornell', 'wisconsin','texas']:
        dataset = WebKB(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['roman-empire', 'amazon-ratings']:
        dataset = HeterophilousGraphDataset(root=f'data/{args.dataset}', name=args.dataset, transform=transforms)
    # Handle NPZ heterophilic datasets
    elif args.dataset.lower() in map(str.lower, get_heterophilic_datasets()):
        # Load the NPZ dataset and convert to a PyG dataset
        pyg_data = load_npz_dataset(args.dataset)
        
        # Create a dummy dataset-like object to maintain compatibility with the rest of the code
        class DummyDataset:
            def __init__(self, data):
                self.data = data
                self.num_classes = data.num_classes
                
            def __getitem__(self, idx):
                return self.data
                
            def __len__(self):
                return 1
                
        dataset = DummyDataset(pyg_data)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset._data_list = None
    return dataset

def get_model(model_type, num_features, num_classes, hidden_dim, num_layers, gat_heads=4):
    """Create a new model instance based on specified type"""
    if model_type.lower() == 'gcn':
        return NodeGCN(num_features, num_classes, hidden_dim, num_layers)
    elif model_type.lower() == 'gat':
        return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads)
    elif model_type.lower() == 'graphconv':
        return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers)   
    elif model_type.lower() == 'graphsage':
        return NodeGraphSAGE(num_features, num_classes, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits without shuffling to preserve natural ordering.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    
    # Calculate sizes
    num_nodes = len(train_indices)
    shared_size = int(0.50 * num_nodes)
    remaining = num_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    original_candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    original_independent_idx = train_indices[shared_size + split_size:shared_size + split_size * 2].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, original_independent_idx, original_candidate_idx
    else:
        return shared_idx, original_candidate_idx, original_independent_idx

def verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger):
    """Verify there is no direct overlap between candidate and independent sets"""
    # Convert to sets for easy comparison
    candidate_set = set(candidate_idx)
    independent_set = set(independent_idx)
    
    # Check: No overlap between candidate and independent sets
    overlap = candidate_set.intersection(independent_set)
    if overlap:
        raise ValueError(f"Data leakage detected! Found {len(overlap)} nodes in both candidate and independent sets")
    
    logger.info("\nData Leakage Check:")
    logger.info(f"✓ No overlap between candidate and independent sets")

def train_models(args, data, shared_idx, candidate_idx, independent_idx, device, logger, output_dir, seeds):
    """Train model f and g on their respective node sets"""
    
    # Create train masks for model f and g
    train_mask_f = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_f[shared_idx + candidate_idx] = True
    
    train_mask_g = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_g[shared_idx + independent_idx] = True
    
    # Print training set information
    logger.info("\nTraining Set Information:")
    logger.info(f"Model f training nodes: {train_mask_f.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Candidate nodes: {len(candidate_idx)}")
    
    logger.info(f"\nModel g training nodes: {train_mask_g.sum().item()}")
    logger.info(f"- Shared nodes: {len(shared_idx)}")
    logger.info(f"- Independent nodes: {len(independent_idx)}")
    
    # Get number of classes
    num_classes = data.y.max().item() + 1
    
    # Lists to store models and their accuracies
    f_models = []
    g_models = []
    f_val_accs = []
    g_val_accs = []
    f_test_accs = []  # Add list for test accuracies
    g_test_accs = []  # Add list for test accuracies
    
    # Seeds for multiple training runs
    training_seeds = seeds
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Input Features: {data.x.size(1)}")
    logger.info(f"Hidden Dimensions: {args.hidden_dim}")
    logger.info(f"Number of Layers: {args.num_layers}")
    logger.info(f"LR: {args.lr}")
    logger.info(f"Weight Decay: {args.weight_decay}")

    if args.model_type == 'gat':
        logger.info(f"Number of Attention Heads: {args.gat_heads}")
    logger.info(f"Output Classes: {num_classes}")
    logger.info(f"Training with seeds: {training_seeds}")
    
    # Track training time
    training_times = []
    
    # Train multiple models with different seeds
    for seed in training_seeds:
        set_seed(seed)
        
        logger.info(f"\nTraining with seed {seed}")
        
        # Start timing for this seed's training
        seed_start_time = time.time()
        
        # Initialize models
        model_f = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        model_g = get_model(args.model_type, data.x.size(1), num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        
        opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_f_val_acc = 0
        best_g_val_acc = 0
        best_f_state = None
        best_g_state = None
        best_f_test_acc = 0
        best_g_test_acc = 0

        for epoch in range(args.epochs):
            # Train model f on shared+candidate nodes
            f_loss = train(model_f, data.x, data.edge_index, 
                         train_mask_f, data.y, opt_f, device)
            f_val_acc = test(model_f, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_f_test_acc = test(model_f, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Train model g on shared+independent nodes
            g_loss = train(model_g, data.x, data.edge_index, 
                         train_mask_g, data.y, opt_g, device)
            g_val_acc = test(model_g, data.x, data.edge_index, 
                           data.val_mask, data.y, device)
            current_g_test_acc = test(model_g, data.x, data.edge_index, 
                                    data.test_mask, data.y, device)
            
            # Save best models based on validation accuracy
            if f_val_acc > best_f_val_acc:
                best_f_val_acc = f_val_acc
                best_f_test_acc = current_f_test_acc
                best_f_state = model_f.state_dict()
            
            if g_val_acc > best_g_val_acc:
                best_g_val_acc = g_val_acc
                best_g_test_acc = current_g_test_acc
                best_g_state = model_g.state_dict()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f'Seed {seed}, Epoch {epoch+1}/{args.epochs}:')
                logger.info(f'Model f - Loss: {f_loss:.4f}, Val Acc: {f_val_acc:.4f}, Test Acc: {current_f_test_acc:.4f}')
                logger.info(f'Model g - Loss: {g_loss:.4f}, Val Acc: {g_val_acc:.4f}, Test Acc: {current_g_test_acc:.4f}')
        
        # Load best states
        model_f.load_state_dict(best_f_state)
        model_g.load_state_dict(best_g_state)
        
        # Store models and accuracies
        f_models.append(model_f.state_dict())
        g_models.append(model_g.state_dict())
        f_val_accs.append(best_f_val_acc)
        g_val_accs.append(best_g_val_acc)
        f_test_accs.append(best_f_test_acc)  # Store test accuracy
        g_test_accs.append(best_g_test_acc)  # Store test accuracy
        
        logger.info(f"\nSeed {seed} Results:")
        logger.info(f"Best Model f - Val Acc: {best_f_val_acc:.4f}, Test Acc: {best_f_test_acc:.4f}")
        logger.info(f"Best Model g - Val Acc: {best_g_val_acc:.4f}, Test Acc: {best_g_test_acc:.4f}")
        
        # Record training time for this seed
        seed_end_time = time.time()
        seed_training_time = seed_end_time - seed_start_time
        training_times.append(seed_training_time)
        logger.info(f"Training time for seed {seed}: {seed_training_time:.2f} seconds")
    
    # Calculate and log average training time
    avg_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)
    logger.info(f"\n===== Training Runtime Analysis =====")
    logger.info(f"Average training time per seed: {avg_training_time:.2f} ± {std_training_time:.2f} seconds")
    logger.info(f"Total training time for {len(training_seeds)} seeds: {sum(training_times):.2f} seconds")
    
    # Select models with best validation accuracy
    f_best_idx = np.argmax(f_val_accs)
    g_best_idx = np.argmax(g_val_accs)
    
    # Save best models
    save_dir = output_dir  # Use the provided output directory
    
    torch.save(f_models[f_best_idx], os.path.join(save_dir, 'f_model.pt'))
    torch.save(g_models[g_best_idx], os.path.join(save_dir, 'g_model.pt'))
    
    logger.info("\nSaved models with best validation accuracy:")
    logger.info(f"Model f - Val Acc: {f_val_accs[f_best_idx]:.4f}")
    logger.info(f"Model g - Val Acc: {g_val_accs[g_best_idx]:.4f}")
    
    # Load best states into models
    model_f = get_model(args.model_type, data.x.size(1), num_classes, 
                     args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    model_g = get_model(args.model_type, data.x.size(1), num_classes, 
                     args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    
    model_f.load_state_dict(f_models[f_best_idx])
    model_g.load_state_dict(g_models[g_best_idx])
    
    # Return best models along with validation accuracies, test accuracy lists, and all model dictionaries
    return model_f, model_g, f_val_accs[f_best_idx], g_val_accs[f_best_idx], f_test_accs, g_test_accs, f_models, g_models, training_times

def perform_memorization_statistical_tests(node_scores, logger):
    """
    Perform statistical tests to check if memorization scores are statistically significant.
    
    Args:
        node_scores: Dictionary of memorization scores by node type
        logger: Logger to output results
    """
    logger.info("\n===== Statistical Significance Tests =====")
    
    # Check if all required node types exist
    required_types = ['candidate', 'shared', 'independent', 'extra']
    for node_type in required_types:
        if node_type not in node_scores:
            logger.info(f"Skipping some statistical tests: '{node_type}' nodes not found")
    
    # 1. Candidate vs other node types
    if 'candidate' in node_scores:
        candidate_scores = node_scores['candidate']['mem_scores']
        
        # Test against each other node type
        for other_type in ['shared', 'independent', 'extra']:
            if other_type not in node_scores:
                continue
                
            other_scores = node_scores[other_type]['mem_scores']
            
            # Run t-test
            t_stat, p_val = stats.ttest_ind(candidate_scores, other_scores, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_diff = np.mean(candidate_scores) - np.mean(other_scores)
            pooled_std = np.sqrt((np.std(candidate_scores)**2 + np.std(other_scores)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_size_interp = "negligible"
            elif effect_size < 0.5:
                effect_size_interp = "small"
            elif effect_size < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
                
            # Interpret p-value
            significant = p_val < 0.01
            
            # Log results
            logger.info(f"\nCandidate vs {other_type} nodes:")
            logger.info(f"  T-statistic: {t_stat:.4f}")
            logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
            logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
            logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 2. Shared vs Independent nodes
    if 'shared' in node_scores and 'independent' in node_scores:
        shared_scores = node_scores['shared']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(shared_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(shared_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(shared_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nShared vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")
    
    # 3. Extra vs Independent nodes
    if 'extra' in node_scores and 'independent' in node_scores:
        extra_scores = node_scores['extra']['mem_scores']
        independent_scores = node_scores['independent']['mem_scores']
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(extra_scores, independent_scores, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(extra_scores) - np.mean(independent_scores)
        pooled_std = np.sqrt((np.std(extra_scores)**2 + np.std(independent_scores)**2) / 2)
        effect_size = abs(mean_diff) / pooled_std
        
        # Interpret effect size
        if effect_size < 0.2:
            effect_size_interp = "negligible"
        elif effect_size < 0.5:
            effect_size_interp = "small"
        elif effect_size < 0.8:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
            
        # Interpret p-value
        significant = p_val < 0.01
        
        # Log results
        logger.info(f"\nExtra vs Independent nodes:")
        logger.info(f"  T-statistic: {t_stat:.4f}")
        logger.info(f"  P-value: {p_val:.6f} ({'significant' if significant else 'not significant'} at p<0.01)")
        logger.info(f"  Effect size (Cohen's d): {effect_size:.4f} ({effect_size_interp})")
        logger.info(f"  Mean difference: {mean_diff:.4f}")

def calculate_graph_metrics(data):
    """Calculate adjusted homophily and node label informativeness for the graph"""
    # Get graph and labels in correct format
    graph, labels = get_graph_and_labels_from_pyg_dataset(data)
    
    # Calculate adjusted homophily
    adj_homophily = h_adj(graph, labels)
    
    # Calculate node label informativeness
    nli = li_node(graph, labels)
    
    return adj_homophily, nli

def log_results_to_csv(args, adj_homophily, nli, test_acc_mean, test_acc_std):
    """Log results to a CSV file"""
    csv_file = 'experiment_results.csv'
    file_exists = os.path.exists(csv_file)
    
    # Prepare the results row
    results = {
        'Dataset': args.dataset,
        'Model': args.model_type,
        'Test_Accuracy_Mean': f"{test_acc_mean:.4f}",
        'Test_Accuracy_Std': f"{test_acc_std:.4f}",
        'Learning_Rate': args.lr,
        'Weight_Decay': args.weight_decay,
        'Adjusted_Homophily': f"{adj_homophily:.4f}",
        'Node_Label_Informativeness': f"{nli:.4f}",
        'Num_Layers': args.num_layers
    }
    
    # Write to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    return csv_file

def save_memorization_scores(node_scores, log_dir, logger=None):
    """
    Save memorization scores to disk for later use.
    
    Args:
        node_scores: Dictionary containing node memorization scores
        log_dir: Directory to save the scores
        logger: Logger for status updates
    """
    # Save the entire node_scores dictionary as a pickle file
    pickle_path = os.path.join(log_dir, 'node_scores.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(node_scores, f)
    
    # Also save individual node types as CSV files for easier inspection
    for node_type, data in node_scores.items():
        if 'raw_data' in data:
            csv_path = os.path.join(log_dir, f'{node_type}_scores.csv')
            data['raw_data'].to_csv(csv_path, index=False)
    
    if logger:
        logger.info(f"Memorization scores saved to {log_dir}")
        logger.info(f"- Complete data: {os.path.basename(pickle_path)}")
        for node_type in node_scores:
            if 'raw_data' in node_scores[node_type]:
                logger.info(f"- {node_type} scores: {node_type}_scores.csv")

def main():
    parser = argparse.ArgumentParser()
    
    # Get all available heterophilic datasets
    heterophilic_datasets = get_heterophilic_datasets()
    # Combine all dataset choices
    all_datasets = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor', 
                   'Chameleon', 'Squirrel', 'Cornell', 'Wisconsin', 'Texas',
                   'Roman-empire', 'Amazon-ratings'] + heterophilic_datasets
                   
    parser.add_argument('--dataset', type=str, required=True,
                       choices=all_datasets,
                       help='Dataset to use for analysis')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv' , 'graphsage'],
                       help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='Number of attention heads for GAT')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--swap_nodes', action='store_true', 
                       help='Swap candidate and independent nodes')
    parser.add_argument('--num_passes', type=int, default=1,
                       help='Number of forward passes to average for confidence scores')
    parser.add_argument('--k_hops', type=int, default=2,
                       help='Number of hops for local neighborhood in NLI calculation')
    parser.add_argument('--noise_level', type=float, default=1.0,
                      help='Standard deviation of Gaussian noise for reliability analysis (default: 0.1)')
    parser.add_argument('--perturb_ratio', type=float, default=0.05,
                      help='Ratio of labels to perturb for label-based reliability analysis (default: 0.05)')
    parser.add_argument('--label_perturb_epochs', type=int, default=1,
                      help='Number of epochs to train on perturbed labels (default: 1)')
    parser.add_argument('--drop_ratio', type=float, default=1.0,
                      help='Ratio of nodes to exclude in generalization analysis (default: 0.05)')
    
    args = parser.parse_args()
    
    # Setup
    logger, log_dir, timestamp = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define training seeds here so they can be used throughout
    training_seeds = [42, 123, 456]
    
    # Load dataset
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    
    # Calculate graph metrics
    adj_homophily, nli = calculate_graph_metrics(data)
    logger.info("\nGraph Metrics:")
    logger.info(f"Adjusted Homophily: {adj_homophily:.4f}")
    logger.info(f"Node Label Informativeness: {nli:.4f}")
    
    # Log dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"Dataset Name: {args.dataset}")
    logger.info(f"Number of Nodes: {data.num_nodes}")
    logger.info(f"Number of Edges: {data.edge_index.size(1)}")
    logger.info(f"Number of Features: {data.num_features}")
    logger.info(f"Number of Classes: {dataset.num_classes}")
    
    # Get node splits
    shared_idx, candidate_idx, independent_idx = get_node_splits(
        data, data.train_mask, swap_candidate_independent=args.swap_nodes
    )
    
    # Get extra indices from test set
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()  # Take first extra_size test indices

    logger.info("\nPartition Statistics:")
    if args.swap_nodes:
        logger.info("Note: Candidate and Independent nodes have been swapped!")
        logger.info("Original independent nodes are now being used as candidate nodes")
        logger.info("Original candidate nodes are now being used as independent nodes")
    logger.info(f"Total train nodes: {data.train_mask.sum().item()}")
    logger.info(f"Shared: {len(shared_idx)} nodes")
    logger.info(f"Candidate: {len(candidate_idx)} nodes")
    logger.info(f"Independent: {len(independent_idx)} nodes")
    logger.info(f"Extra test nodes: {len(extra_indices)} nodes")
    logger.info(f"Val set: {data.val_mask.sum().item()} nodes")
    logger.info(f"Test set: {data.test_mask.sum().item()} nodes")
    
    # Create nodes_dict
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices,
        'val': torch.where(data.val_mask)[0].tolist(),
        'test': torch.where(data.test_mask)[0].tolist()
    }
    
    # Verify no data leakage
    verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger)
    
    # Initialize test accuracy tracking
    f_test_accs = []
    g_test_accs = []
    
    # Train models
    logger.info("\nTraining models...")
    model_f, model_g, f_val_acc, g_val_acc, f_test_accs, g_test_accs, f_models, g_models, model_training_times = train_models(
        args=args,
        data=data,
        shared_idx=shared_idx,
        candidate_idx=candidate_idx,
        independent_idx=independent_idx,
        device=device,
        logger=logger,
        output_dir=log_dir,
        seeds=training_seeds  # Pass seeds to train_models
    )
    
    # Get test accuracy statistics
    avg_f_test_acc = np.mean(f_test_accs)
    std_f_test_acc = np.std(f_test_accs)
    
    # Calculate memorization scores for the best model
    logger.info("\nCalculating memorization scores for best model...")
    best_model_scores = calculate_node_memorization_score(
        model_f=model_f,
        model_g=model_g,
        data=data,
        nodes_dict=nodes_dict,
        device=device,
        logger=logger,
        num_passes=args.num_passes
    )
    
    # Save memorization scores to disk
    save_memorization_scores(best_model_scores, log_dir, logger)
    
    # Calculate and log average scores for each node type
    for node_type, scores_dict in best_model_scores.items():
        logger.info(f"Best model memorization score for {node_type} nodes: {scores_dict['avg_score']:.4f}")
        # Also log the number of nodes above threshold
        logger.info(f"Nodes with score > 0.5: {scores_dict['nodes_above_threshold']}/{len(scores_dict['mem_scores'])} ({scores_dict['percentage_above_threshold']:.1f}%)")
    
    # Perform statistical tests on best model's memorization scores
    perform_memorization_statistical_tests(best_model_scores, logger)
    
    # Create visualization for best model scores
    plot_filename = f'{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}_best.pdf'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_node_memorization_analysis(
        node_scores=best_model_scores,
        save_path=plot_path,
        title_suffix=f"Best Model | Dataset: {args.dataset}, Model: {args.model_type}\nf_acc={f_val_acc:.3f}, g_acc={g_val_acc:.3f}",
        node_types_to_plot=['shared', 'candidate', 'independent', 'extra']
    )
    logger.info(f"Best model memorization score plot saved to: {plot_path}")
    
    # Now calculate scores for all seed models and average them
    logger.info("\nCalculating memorization scores for all seed models...")
    
    # Dictionary to store results for each seed
    all_seed_scores = {seed: {} for seed in training_seeds}
    
    # Dictionary to store average results across seeds
    avg_node_scores = {}
    
    # Track time specifically for candidate node memorization score calculation
    candidate_memorization_times = []
    
    for i, seed in enumerate(training_seeds):
        logger.info(f"\nCalculating scores for seed {seed}...")
        
        # Create fresh models with the seed's weights
        seed_model_f = get_model(args.model_type, data.x.size(1), dataset.num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        seed_model_g = get_model(args.model_type, data.x.size(1), dataset.num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        
        # Load the trained weights for this seed
        seed_model_f.load_state_dict(f_models[i])
        seed_model_g.load_state_dict(g_models[i])
        
        # Calculate scores for this seed
        seed_scores = calculate_node_memorization_score(
            model_f=seed_model_f,
            model_g=seed_model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=None,  # Avoid verbose logging for each seed
            num_passes=args.num_passes
        )
        
        # Time specifically for candidate nodes
        if 'candidate' in seed_scores:
            # Measure time for calculating candidate node memorization scores only
            # Since the calculation is already done, we'll measure a focused recalculation
            # just for candidate nodes
            start_time = time.time()
            
            candidate_scores = calculate_node_memorization_score(
                model_f=seed_model_f,
                model_g=seed_model_g,
                data=data,
                nodes_dict={'candidate': nodes_dict['candidate']},  # Only candidate nodes
                device=device,
                logger=None,
                num_passes=args.num_passes
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            candidate_memorization_times.append(elapsed_time)
            logger.info(f"Candidate node memorization score calculation for seed {seed} took {elapsed_time:.2f} seconds")
        
        # Store the scores for this seed
        all_seed_scores[training_seeds[i]] = seed_scores
    
    # Calculate average scores and 95% confidence intervals across seeds
    for node_type in nodes_dict.keys():
        if node_type in ['val', 'test']:
            continue
            
        # Check if this node type exists in all seed results
        if not all(node_type in all_seed_scores[seed] for seed in training_seeds):
            logger.warning(f"Node type {node_type} not present in all seed results, skipping averaging")
            continue
        
        # Collect scores across all seeds
        all_mem_scores = []
        all_f_confidences = []
        all_g_confidences = []
        all_above_threshold_counts = []
        all_above_threshold_percentages = []
        
        for seed in training_seeds:
            all_mem_scores.append(all_seed_scores[seed][node_type]['mem_scores'])
            all_f_confidences.append(all_seed_scores[seed][node_type]['f_confidences'])
            all_g_confidences.append(all_seed_scores[seed][node_type]['g_confidences'])
            all_above_threshold_counts.append(all_seed_scores[seed][node_type]['nodes_above_threshold'])
            all_above_threshold_percentages.append(all_seed_scores[seed][node_type]['percentage_above_threshold'])
        
        # Calculate mean and 95% confidence intervals across seeds
        mean_mem_scores = np.mean([np.mean(scores) for scores in all_mem_scores])
        ci_mem_scores = 1.96 * np.std([np.mean(scores) for scores in all_mem_scores]) / np.sqrt(len(training_seeds))
        
        mean_above_threshold = np.mean(all_above_threshold_counts)
        ci_above_threshold = 1.96 * np.std(all_above_threshold_counts) / np.sqrt(len(training_seeds))
        
        mean_percentage_above = np.mean(all_above_threshold_percentages)
        ci_percentage_above = 1.96 * np.std(all_above_threshold_percentages) / np.sqrt(len(training_seeds))
        
        # Store average results
        avg_node_scores[node_type] = {
            'mem_scores': best_model_scores[node_type]['mem_scores'],  # Use best model's detailed scores for plotting
            'f_confidences': best_model_scores[node_type]['f_confidences'],
            'g_confidences': best_model_scores[node_type]['g_confidences'],
            'avg_score': mean_mem_scores,
            'ci_score': ci_mem_scores,
            'nodes_above_threshold': mean_above_threshold,
            'ci_above_threshold': ci_above_threshold,
            'percentage_above_threshold': mean_percentage_above,
            'ci_percentage_above': ci_percentage_above,
            'all_seed_above_threshold': all_above_threshold_counts,
            'raw_data': best_model_scores[node_type]['raw_data']  # Keep best model's raw data for compatibility
        }
        
        # Log average statistics
        logger.info(f"\nAverage statistics for {node_type} nodes across {len(training_seeds)} seeds:")
        logger.info(f"  Mean memorization score: {mean_mem_scores:.4f} ± {ci_mem_scores:.4f} (95% CI)")
        logger.info(f"  Nodes with score > 0.5: {mean_above_threshold:.1f} ± {ci_above_threshold:.1f} ({mean_percentage_above:.1f}% ± {ci_percentage_above:.1f}%)")
    
    # Create a new visualization for average scores across all seeds
    # This plot focuses on the frequency of memorized nodes with confidence intervals
    avg_plot_filename = f'{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}_avg.png'
    avg_plot_path = os.path.join(log_dir, avg_plot_filename)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Colors and category names
    colors = {'candidate': 'blue', 'independent': 'orange', 'extra': 'green', 'shared': 'red'}
    labels = {'candidate': '$S_C$', 'independent': '$S_I$', 'extra': '$S_E$', 'shared': '$S_S$'}
    
    # Prepare data for plotting
    node_types = [nt for nt in ['shared', 'candidate', 'independent', 'extra'] if nt in avg_node_scores]
    x_pos = np.arange(len(node_types))
    memorized_percentages = [avg_node_scores[nt]['percentage_above_threshold'] for nt in node_types]
    ci_percentages = [avg_node_scores[nt]['ci_percentage_above'] for nt in node_types]
    bar_colors = [colors[nt] for nt in node_types]
    
    # Create bar plot with error bars
    plt.bar(x_pos, memorized_percentages, yerr=ci_percentages, capsize=10, color=bar_colors, alpha=0.7)
    
    # Add count labels on top of bars
    for i, nt in enumerate(node_types):
        count = avg_node_scores[nt]['nodes_above_threshold']
        ci = avg_node_scores[nt]['ci_above_threshold']
        total = len(best_model_scores[nt]['mem_scores'])
        plt.text(i, memorized_percentages[i] + ci_percentages[i] + 2, 
                f"{count:.1f}±{ci:.1f}/{total}", 
                ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Node Type', fontsize=20, font='Sans Serif')
    plt.ylabel('Percentage of Memorized Nodes (%)', fontsize=20, font='Sans Serif')
    #plt.title(f'Average Percentage of Memorized Nodes Across {len(training_seeds)} Seeds\nDataset: {args.dataset}, Model: {args.model_type}', fontsize=14)
    plt.xticks(x_pos, [labels[nt] for nt in node_types])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a line for the memorization threshold
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Threshold = 0.5')
    
    plt.tight_layout()
    plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Average memorization frequency plot saved to: {avg_plot_path}")
    
    # Report average memorization time for candidate nodes
    avg_candidate_memorization_time = np.mean(candidate_memorization_times)
    std_candidate_memorization_time = np.std(candidate_memorization_times)
    logger.info(f"\n===== Candidate Node Memorization Score Runtime Analysis =====")
    logger.info(f"Average time to calculate candidate node memorization scores: {avg_candidate_memorization_time:.2f} ± {std_candidate_memorization_time:.2f} seconds")
    logger.info(f"Total time for all {len(training_seeds)} seeds: {sum(candidate_memorization_times):.2f} seconds")
    
    # Calculate and report the combined training + memorization time
    combined_times = []
    for i in range(len(training_seeds)):
        combined_time = model_training_times[i] + candidate_memorization_times[i]
        combined_times.append(combined_time)
    
    avg_combined_time = np.mean(combined_times)
    std_combined_time = np.std(combined_times)
    logger.info(f"\n===== Combined Training and Memorization Runtime Analysis =====")
    logger.info(f"Final average over {len(training_seeds)} random seeds (training f and g + memorization score calculation for candidate nodes): {avg_combined_time:.2f} ± {std_combined_time:.2f} seconds")
    logger.info(f"Total combined time for all {len(training_seeds)} seeds: {sum(combined_times):.2f} seconds")
    
    # Run kNN label disagreement analysis
    logger.info("\n===== Running kNN Label Disagreement Analysis =====")
    
    # Structure to store kNN disagreement results for each seed
    all_seed_knn_results = {}
    all_memorized_means = []
    all_non_memorized_means = []
    all_disagreement_diffs = []
    
    # Track time for kNN label disagreement calculation
    knn_times = []
    
    # Run kNN label disagreement analysis for each seed
    for i, seed in enumerate(training_seeds):
        logger.info(f"\nRunning kNN label disagreement analysis for seed {seed}...")
        
        # Create a seed-specific timestamp suffix for file naming
        seed_timestamp = f"{timestamp}_seed{seed}"
        
        # Measure time for kNN label disagreement analysis
        start_time = time.time()
        
        # Run the analysis for this seed's model scores
        seed_knn_results = run_knn_label_disagreement_analysis(
            data=data,
            nodes_dict=nodes_dict,
            node_scores=all_seed_scores[seed],
            save_dir=log_dir,
            model_type=args.model_type,
            dataset_name=args.dataset,
            timestamp=seed_timestamp,
            k=3,  # Default k value - ensure this matches in all places
            threshold=0.5,  # Default threshold for memorization
            similarity_metric='euclidean',  # Default similarity metric
            device=device,
            logger=None  # Avoid verbose logging for each seed
        )
        
        # Record the time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        knn_times.append(elapsed_time)
        logger.info(f"kNN label disagreement analysis for seed {seed} took {elapsed_time:.2f} seconds")
        
        # Store the results for this seed
        all_seed_knn_results[seed] = seed_knn_results
        
        # Extract key metrics for statistical analysis
        default_analysis = seed_knn_results['default']['analysis']
        mem_mean = default_analysis['memorized']['mean']
        non_mem_mean = default_analysis['non_memorized']['mean']
        
        # Store for calculating average and CI
        all_memorized_means.append(mem_mean)
        all_non_memorized_means.append(non_mem_mean)
        all_disagreement_diffs.append(mem_mean - non_mem_mean)
    
    # Calculate averages and 95% confidence intervals
    avg_memorized_mean = np.mean(all_memorized_means)
    avg_non_memorized_mean = np.mean(all_non_memorized_means)
    avg_diff = np.mean(all_disagreement_diffs)
    
    ci_memorized = 1.96 * np.std(all_memorized_means) / np.sqrt(len(training_seeds))
    ci_non_memorized = 1.96 * np.std(all_non_memorized_means) / np.sqrt(len(training_seeds))
    ci_diff = 1.96 * np.std(all_disagreement_diffs) / np.sqrt(len(training_seeds))
    
    # Calculate average kNN runtime
    avg_knn_time = np.mean(knn_times)
    std_knn_time = np.std(knn_times)
    
    # Log the runtime results
    logger.info("\n===== kNN Label Disagreement Runtime Analysis =====")
    logger.info(f"Average time to calculate kNN label disagreement: {avg_knn_time:.2f} ± {std_knn_time:.2f} seconds")
    logger.info(f"Total time for all {len(training_seeds)} seeds: {sum(knn_times):.2f} seconds")
    
    # Perform statistical tests on the averaged results
    # 1. Paired t-test (because we're comparing scores from the same models)
    t_stat, p_val_t = stats.ttest_rel(all_memorized_means, all_non_memorized_means)
    
    # 2. Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    try:
        wilcoxon_stat, p_val_wilcoxon = stats.wilcoxon(all_memorized_means, all_non_memorized_means)
    except ValueError as e:
        # Handle case where sample size is too small
        wilcoxon_stat, p_val_wilcoxon = None, None
        logger.info(f"Could not perform Wilcoxon test: {e}")
    
    # Effect size calculation (Cohen's d for paired samples)
    d = avg_diff / np.std(all_disagreement_diffs)
    
    # Interpret effect size
    if abs(d) < 0.2:
        effect_size_interp = "negligible"
    elif abs(d) < 0.5:
        effect_size_interp = "small"
    elif abs(d) < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"
    
    # Log the averaged results
    logger.info("\n===== Average kNN Label Disagreement Results (across all seeds) =====")
    logger.info(f"Memorized nodes: mean={avg_memorized_mean:.4f} ± {ci_memorized:.4f} (95% CI)")
    logger.info(f"Non-memorized nodes: mean={avg_non_memorized_mean:.4f} ± {ci_non_memorized:.4f} (95% CI)")
    logger.info(f"Mean difference: {avg_diff:.4f} ± {ci_diff:.4f} (95% CI)")
    
    # Log statistical test results
    logger.info("\n===== Statistical Significance Tests for kNN Disagreement =====")
    logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_val_t:.6f} ({'significant' if p_val_t < 0.01 else 'not significant'} at p<0.01)")
    if p_val_wilcoxon is not None:
        logger.info(f"Wilcoxon signed-rank test: W={wilcoxon_stat:.4f}, p={p_val_wilcoxon:.6f} ({'significant' if p_val_wilcoxon < 0.01 else 'not significant'} at p<0.01)")
    logger.info(f"Effect size (Cohen's d): {d:.4f} ({effect_size_interp})")
    
    # Also run once with the best model for visualization purposes
    logger.info("\nRunning kNN label disagreement analysis with best model (for visualization)...")
    best_knn_results = run_knn_label_disagreement_analysis(
        data=data,
        nodes_dict=nodes_dict,
        node_scores=best_model_scores,
        save_dir=log_dir,
        model_type=args.model_type,
        dataset_name=args.dataset,
        timestamp=timestamp,
        k=3,  # Default k value
        threshold=0.5,  # Default threshold for memorization
        similarity_metric='euclidean',  # Default similarity metric
        device=device,
        logger=logger
    )
    
    # # Create detailed visualization of the kNN disagreement process
    # logger.info("\nCreating kNN disagreement process visualization...")
    # visualize_knn_disagreement_process(
    #     data=data,
    #     nodes_dict=nodes_dict,
    #     node_scores=best_model_scores,
    #     save_dir=log_dir,
    #     model_type=args.model_type,
    #     dataset_name=args.dataset,
    #     timestamp=timestamp,
    #     k=10,  # Default k value
    #     threshold=0.5,  # Default threshold
    #     similarity_metric='euclidean',  # Default similarity metric
    #     num_example_nodes=4,  # Number of example nodes to show in detail
    #     device=device,
    #     logger=logger,
    #     embedding_method='tsne',  # Method for dimensionality reduction
    #     random_seed=42  # For reproducibility
    # )

if __name__ == "__main__":
    main()
