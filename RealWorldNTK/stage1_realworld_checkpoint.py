import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
import gc
import time
from tqdm import tqdm

from main_fixed import (
    set_seed, 
    get_model, 
    load_dataset, 
    train, 
    test, 
    get_node_splits, 
    verify_no_data_leakage
)
from memorization import calculate_node_memorization_score

def setup_logging(args, stage="stage1"):
    """Set up logging directory and file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with descriptive structure
    dir_name = f"{args.model_type}_{args.dataset}_{args.num_layers}_{timestamp}"
    
    # Create base results directory if it doesn't exist
    base_dir = 'results_realworld'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create full directory path
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger(f'{stage}_realworld')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    log_file = os.path.join(log_dir, f'{stage}_{args.model_type}_{args.dataset}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    logger.info(f"Set up logging to {log_file}")
    
    return logger, log_dir, checkpoint_dir, timestamp

def train_model_to_checkpoint(args, model, data, train_mask, epoch, device, logger, optimizer=None):
    """Train a model to a specific epoch checkpoint."""
    model.to(device)
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    logger.info(f"Training model to epoch {epoch}...")
    model.train()
    for e in range(epoch):
        loss = train(model, data.x, data.edge_index, train_mask, data.y, optimizer, device)
        
        # Optionally print training progress
        if (e + 1) % 10 == 0 or e == epoch - 1:
            val_acc = test(model, data.x, data.edge_index, data.val_mask, data.y, device)
            logger.info(f"Epoch {e+1}/{epoch}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final evaluation
    model.eval()
    val_acc = test(model, data.x, data.edge_index, data.val_mask, data.y, device)
    test_acc = test(model, data.x, data.edge_index, data.test_mask, data.y, device)
    logger.info(f"Checkpoint at epoch {epoch} - Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return model, val_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='Stage 1: Train GNN models with checkpoints')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use (e.g., Cora, Citeseer, Pubmed, etc.)')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'graphconv', 'graphsage'],
                        help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Number of attention heads for GAT')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs to train')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[10, 25, 50, 100],
                        help='Epochs at which to save checkpoints')
    parser.add_argument('--swap_nodes', action='store_true',
                        help='Swap candidate and independent nodes')
    
    # Memorization calculation
    parser.add_argument('--num_passes', type=int, default=1,
                        help='Number of forward passes to average for confidence scores')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger, log_dir, checkpoint_dir, timestamp = setup_logging(args)
    logger.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    
    # Log dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"Dataset Name: {args.dataset}")
    logger.info(f"Number of Nodes: {data.num_nodes}")
    logger.info(f"Number of Edges: {data.edge_index.size(1)}")
    logger.info(f"Number of Features: {data.num_features if hasattr(data, 'num_features') else data.x.size(1)}")
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
    
    # Create train masks for model f and g
    train_mask_f = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_f[shared_idx + candidate_idx] = True
    
    train_mask_g = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_g[shared_idx + independent_idx] = True
    
    # Number of classes
    num_classes = dataset.num_classes
    
    # Initialize results container
    temporal_results = []
    
    # Create optimizers
    model_f = get_model(args.model_type, data.x.size(1), num_classes, 
                        args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    model_g = get_model(args.model_type, data.x.size(1), num_classes, 
                        args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    
    optimizer_f = optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_g = optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Ensure checkpoints are sorted and unique
    checkpoints = sorted(list(set(args.checkpoints)))
    
    # Track previous checkpoint to avoid retraining from scratch
    prev_checkpoint = 0
    
    # Loop through checkpoints
    for checkpoint in tqdm(checkpoints, desc="Processing checkpoints"):
        logger.info(f"\n{'='*30}\nProcessing checkpoint at epoch {checkpoint}\n{'='*30}")
        
        # Calculate epochs to train (incremental)
        epochs_to_train = checkpoint - prev_checkpoint
        
        # Train models incrementally to checkpoint
        logger.info(f"Training models for additional {epochs_to_train} epochs...")
        
        # Train model f for epochs_to_train epochs
        model_f, f_val_acc, f_test_acc = train_model_to_checkpoint(
            args, model_f, data, train_mask_f, epochs_to_train, device, logger, optimizer_f
        )
        
        # Train model g for epochs_to_train epochs
        model_g, g_val_acc, g_test_acc = train_model_to_checkpoint(
            args, model_g, data, train_mask_g, epochs_to_train, device, logger, optimizer_g
        )
        
        # Save checkpoints
        f_checkpoint_path = os.path.join(checkpoint_dir, f"model_f_epoch{checkpoint}.pt")
        g_checkpoint_path = os.path.join(checkpoint_dir, f"model_g_epoch{checkpoint}.pt")
        
        torch.save(model_f.state_dict(), f_checkpoint_path)
        torch.save(model_g.state_dict(), g_checkpoint_path)
        logger.info(f"Saved model checkpoints for epoch {checkpoint}")
        
        # Calculate memorization scores
        logger.info("Calculating memorization scores...")
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=None,  # Avoid verbose logging
            num_passes=args.num_passes
        )
        
        # Extract memorization rate for candidate nodes
        if 'candidate' in node_scores:
            mem_rate = node_scores['candidate']['percentage_above_threshold']
            logger.info(f"Memorization rate for candidate nodes at epoch {checkpoint}: {mem_rate:.2f}%")
        else:
            mem_rate = -1
            logger.warning("No candidate nodes found in memorization scores")
        
        # Store results
        result = {
            'epoch': checkpoint,
            'f_val_acc': f_val_acc,
            'f_test_acc': f_test_acc,
            'g_val_acc': g_val_acc,
            'g_test_acc': g_test_acc,
            'memorization_rate': mem_rate
        }
        temporal_results.append(result)
        
        # Update prev_checkpoint
        prev_checkpoint = checkpoint
    
    # Save results to CSV
    results_df = pd.DataFrame(temporal_results)
    csv_path = os.path.join(log_dir, f'stage1_results_{args.dataset}_{args.model_type}.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved stage1 results to {csv_path}")
    
    # Plot memorization rate over time
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['epoch'], results_df['memorization_rate'], 'o-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Memorization Rate (%)')
    plt.title(f'Memorization Rate Over Time - {args.dataset}, {args.model_type.upper()}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plot_path = os.path.join(log_dir, f'memorization_rate_{args.dataset}_{args.model_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved memorization rate plot to {plot_path}")
    
    logger.info("Stage 1 analysis complete!")

if __name__ == '__main__':
    main()
