import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
import gc
import json
from tqdm import tqdm
import traceback

from main_fixed import (
    set_seed, 
    get_model, 
    load_dataset
)
from metrics import empirical_ntk, centered_kernel_alignment

def setup_logging(args, stage="stage2"):
    """Set up logging directory and file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory name with descriptive structure
    dir_name = f"{args.model_type}_{args.dataset}_{args.stage1_seed}_{timestamp}"
    
    # Create base results directory if it doesn't exist
    base_dir = 'results_realworld_ntk'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create full directory path
    log_dir = os.path.join(base_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
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
    
    return logger, log_dir, timestamp

def find_stage1_run_dir(args):
    """Find the relevant stage1 run directory based on dataset, model type, and seed."""
    import glob
    
    # First check if the exact run_info file exists
    run_info_pattern = f"stage1_run_info_{args.dataset}_{args.model_type}_seed{args.stage1_seed}_*.json"
    run_info_files = glob.glob(os.path.join(args.stage1_dir, run_info_pattern))
    
    if run_info_files:
        # Use the most recent run info file
        latest_run_info = sorted(run_info_files)[-1]
        with open(latest_run_info, 'r') as f:
            run_info = json.load(f)
        return run_info['run_dir']
    
    # First fallback: look for stage1_* directories
    run_pattern = f"stage1_{args.dataset}_{args.model_type}_seed{args.stage1_seed}_*"
    run_dirs = glob.glob(os.path.join(args.stage1_dir, run_pattern))
    
    if run_dirs:
        return sorted(run_dirs)[-1]
    
    # Second fallback: look in the stage1_dir itself
    if os.path.exists(os.path.join(args.stage1_dir, "checkpoints")):
        return args.stage1_dir
        
    # Third fallback: more flexible directory matching
    # Try to match the pattern for all directories in the stage1_dir that contain the dataset and model_type
    pattern = f"*{args.dataset}*{args.model_type}*"
    or_pattern = f"*{args.model_type}*{args.dataset}*"
    
    matching_dirs = glob.glob(os.path.join(args.stage1_dir, pattern))
    matching_dirs.extend(glob.glob(os.path.join(args.stage1_dir, or_pattern)))
    
    if matching_dirs:
        return sorted(matching_dirs)[-1]  # Take the most recent matching directory
        
    # If we still can't find it, raise an error with more details
    raise FileNotFoundError(
        f"Could not find stage1 run directory for {args.dataset}, {args.model_type}, seed {args.stage1_seed}.\n"
        f"Searched in: {args.stage1_dir}\n"
        f"Available directories: {os.listdir(args.stage1_dir)}"
    )

def construct_sparse_adj(edge_index, num_nodes, type='DAD', self_loop=True, device=None):
    """Construct sparse adjacency matrix without requiring PyTorch Geometric."""
    edge_index = edge_index.cpu().numpy()
    
    # Convert to undirected if needed (add reverse edges)
    if edge_index.shape[1] > 0:  # If there are edges
        edge_index_rev = np.stack([edge_index[1], edge_index[0]], axis=0)
        # Combine original and reversed edges
        edge_index_combined = np.concatenate([edge_index, edge_index_rev], axis=1)
        
        # Remove duplicates by converting to set of tuples and back
        edge_tuples = set(zip(edge_index_combined[0], edge_index_combined[1]))
        edge_index = np.array(list(edge_tuples)).T
        if edge_index.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
    
    # Add self-loops if requested
    if self_loop:
        # Remove existing self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Add self-loops for all nodes
        self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0)
        edge_index = np.concatenate([edge_index, self_loops], axis=1)
    
    # Convert back to torch tensor
    edge_index = torch.from_numpy(edge_index).long().to(device if device else 'cpu')
    
    # Calculate degrees
    dst_nodes = edge_index[1]
    ones = torch.ones(dst_nodes.size(0), device=dst_nodes.device)
    deg = torch.zeros(num_nodes, device=dst_nodes.device)
    deg.scatter_add_(0, dst_nodes, ones)
    
    # Apply normalization
    src, dst = edge_index
    if type == 'DAD':
        deg_src = deg[src].pow(-0.5)
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = deg[dst].pow(-0.5)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    elif type == 'DA':
        deg_src = deg[src].pow(-1)
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst = torch.ones_like(deg_dst)  # No normalization for dst
    
    edge_values = deg_src * deg_dst
    A = torch.sparse_coo_tensor(edge_index, edge_values, torch.Size([num_nodes, num_nodes]))
    return A

def calculate_ntk_with_oom_handling(model, data, logger, device):
    """Calculate NTK with OOM handling by falling back to CPU if necessary."""
    try:
        logger.info(f"Attempting to calculate NTK on {device}...")
        model.to(device)
        Theta_t = empirical_ntk(model, data.x.to(device), edge_index=data.edge_index.to(device), type='nn')
        return Theta_t
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            logger.warning("CUDA out of memory encountered, falling back to CPU")
            try:
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Move model and data to CPU
                cpu_device = torch.device('cpu')
                model.to(cpu_device)
                
                # Calculate NTK on CPU
                logger.info("Calculating NTK on CPU (this may take a while)...")
                Theta_t = empirical_ntk(model, data.x.cpu(), edge_index=data.edge_index.cpu(), type='nn')
                return Theta_t
            except Exception as cpu_e:
                logger.error(f"Failed to calculate NTK on CPU: {cpu_e}")
                logger.error(traceback.format_exc())
                return None
        else:
            logger.error(f"Error during NTK calculation: {e}")
            logger.error(traceback.format_exc())
            return None

def plot_temporal_dynamics(df, dataset_name, model_type, output_dir, logger):
    """Create visualization of temporal dynamics for memorization rate and NTK alignments."""
    plt.figure(figsize=(12, 10))
    
    # Create 3 subplots stacked vertically
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Plot 1: Memorization Rate over time
    ax1.plot(df['epoch'], df['memorization_rate'], 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_ylabel('Memorization Rate (%)', fontsize=14)
    ax1.set_title(f'Temporal Dynamics for {dataset_name}, {model_type.upper()}', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: NTK-A Alignment over time
    ax2.plot(df['epoch'], df['align_ntk_a'], 's-', color='purple', linewidth=2, markersize=8)
    ax2.set_ylabel('A(Θₜ,A)', fontsize=14, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: NTK-Y Alignment over time
    ax3.plot(df['epoch'], df['align_ntk_y'], 'd-', color='royalblue', linewidth=2, markersize=8)
    ax3.set_ylabel('A(Θₜ,Θ*)', fontsize=14, color='royalblue')
    ax3.set_xlabel('Training Epoch', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='royalblue')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Draw horizontal line for constant Graph-Label alignment
    if 'align_a_y_const' in df.columns:
        align_a_y_const = df['align_a_y_const'].iloc[0]
        ax3.axhline(y=align_a_y_const, color='firebrick', linestyle='--', 
                  label=f'A(A,Θ*) = {align_a_y_const:.4f}')
        ax3.legend(loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'temporal_dynamics_{dataset_name}_{model_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved temporal dynamics plot to {save_path}")
    return save_path

def plot_correlation(df, output_dir, logger):
    """Create scatter plots to visualize correlations between memorization rate and NTK alignments."""
    plt.figure(figsize=(15, 6))
    
    # Plot 1: MR vs NTK-A alignment
    plt.subplot(1, 2, 1)
    sns.regplot(x='align_ntk_a', y='memorization_rate', data=df, scatter_kws={'s': 80})
    plt.xlabel('NTK-Graph Alignment A(Θₜ,A)', fontsize=12)
    plt.ylabel('Memorization Rate (%)', fontsize=12)
    plt.title('Memorization Rate vs NTK-Graph Alignment', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate and add correlation coefficient
    corr_mr_ntk_a = df['memorization_rate'].corr(df['align_ntk_a'])
    plt.annotate(f'r = {corr_mr_ntk_a:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Plot 2: MR vs NTK-Y alignment
    plt.subplot(1, 2, 2)
    sns.regplot(x='align_ntk_y', y='memorization_rate', data=df, scatter_kws={'s': 80})
    plt.xlabel('NTK-Label Alignment A(Θₜ,Θ*)', fontsize=12)
    plt.ylabel('Memorization Rate (%)', fontsize=12)
    plt.title('Memorization Rate vs NTK-Label Alignment', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate and add correlation coefficient
    corr_mr_ntk_y = df['memorization_rate'].corr(df['align_ntk_y'])
    plt.annotate(f'r = {corr_mr_ntk_y:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mr_ntk_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation plot to {save_path}")
    return save_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stage 2: Calculate NTK dynamics from checkpoints')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (must match stage 1)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['gcn', 'gat', 'graphconv', 'graphsage'],
                       help='GNN model type (must match stage 1)')
    parser.add_argument('--stage1_seed', type=int, default=42,
                       help='Random seed used in stage 1')
    parser.add_argument('--stage1_dir', type=str, required=True,
                       help='Directory containing stage 1 output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (falls back to CPU if OOM)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    
    # Override device if force_cpu is set
    if args.force_cpu:
        args.device = 'cpu'
    
    # Setup logging
    logger, log_dir, timestamp = setup_logging(args)
    logger.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Find stage 1 run directory
    try:
        stage1_run_dir = find_stage1_run_dir(args)
        logger.info(f"Found stage 1 run directory: {stage1_run_dir}")
        
        # Locate checkpoint directory with more flexibility
        possible_checkpoint_dirs = [
            os.path.join(stage1_run_dir, "checkpoints"),
            stage1_run_dir,  # In case checkpoints are directly in the run directory
            os.path.join(args.stage1_dir, "checkpoints")  # In case checkpoints are in the parent directory
        ]
        
        checkpoint_dir = None
        for dir_path in possible_checkpoint_dirs:
            if os.path.isdir(dir_path):
                # Additional check: see if this directory contains any model_f* files
                import glob
                if glob.glob(os.path.join(dir_path, "model_f*")):
                    checkpoint_dir = dir_path
                    break
        
        if not checkpoint_dir:
            raise FileNotFoundError(
                f"Checkpoint directory not found. Searched in: {possible_checkpoint_dirs}. "
                f"Available files in run dir: {os.listdir(stage1_run_dir)}"
            )
        
        logger.info(f"Found checkpoint directory: {checkpoint_dir}")
        
        # Load stage 1 results with flexible naming patterns
        stage1_csv_patterns = [
            os.path.join(args.stage1_dir, f"stage1_results_{args.dataset}_{args.model_type}_seed{args.stage1_seed}.csv"),
            os.path.join(stage1_run_dir, f"stage1_results_{args.dataset}_{args.model_type}_seed{args.stage1_seed}.csv"),
            os.path.join(stage1_run_dir, f"stage1_results_{args.dataset}_{args.model_type}.csv"),
            os.path.join(args.stage1_dir, f"stage1_results_{args.dataset}_{args.model_type}.csv")
        ]
        
        stage1_csv = None
        for pattern in stage1_csv_patterns:
            if os.path.exists(pattern):
                stage1_csv = pattern
                break
        
        if not stage1_csv:
            # Try to find any stage1_results CSV file in the run directory
            import glob
            csv_files = glob.glob(os.path.join(stage1_run_dir, "stage1_results_*.csv"))
            if csv_files:
                stage1_csv = sorted(csv_files)[-1]  # Use the most recent CSV
        
        if not stage1_csv:
            raise FileNotFoundError(
                f"Stage 1 results CSV not found. Searched patterns: {stage1_csv_patterns}. "
                f"Available files in run dir: {os.listdir(stage1_run_dir)}"
            )
        
        mr_df = pd.read_csv(stage1_csv)
        logger.info(f"Loaded stage 1 results from {stage1_csv} with {len(mr_df)} checkpoints")
        
    except Exception as e:
        logger.error(f"Error locating stage 1 data: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Load dataset (needs same preprocessing as stage 1)
    try:
        logger.info(f"Loading dataset: {args.dataset}")
        dataset_args = argparse.Namespace()
        dataset_args.dataset = args.dataset
        dataset = load_dataset(dataset_args)
        data = dataset[0]
        
        # Get number of classes
        num_classes = data.y.max().item() + 1
        logger.info(f"Dataset loaded successfully. Number of classes: {num_classes}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Pre-compute graph and ideal kernels
    logger.info("Pre-computing constant kernels (Graph and Label)...")
    
    try:
        # Number of classes
        Y_onehot = F.one_hot(data.y, num_classes=num_classes).float().to(device)
        Theta_star = Y_onehot @ Y_onehot.T
        
        # Graph kernel (A_dense)
        adj_sparse = construct_sparse_adj(data.edge_index, data.num_nodes, device=device)
        A_dense = adj_sparse.to_dense()
        
        # Calculate Graph-Label alignment (constant)
        align_a_y_const = centered_kernel_alignment(A_dense, Theta_star)
        logger.info(f"Pre-computed constant Graph-Label alignment (A-Y): {align_a_y_const:.4f}")
    except Exception as e:
        logger.error(f"Error during pre-computation of constant kernels: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Initialize results container
    ntk_results = []
    
    # Process each checkpoint
    for index, row in mr_df.iterrows():
        t_ckpt = int(row['epoch'])
        logger.info(f"\n--- Processing Checkpoint: Epoch {t_ckpt} ---")
        
        # Find checkpoint file - try multiple possible naming patterns
        checkpoint_patterns = [
            os.path.join(checkpoint_dir, f"model_f_{args.dataset}_{args.model_type}_seed{args.stage1_seed}_epoch{t_ckpt}.pth"),
            os.path.join(checkpoint_dir, f"model_f_epoch{t_ckpt}.pt"),
            os.path.join(checkpoint_dir, f"model_f_epoch{t_ckpt}.pth"),
            os.path.join(checkpoint_dir, f"model_f_{args.dataset}_{args.model_type}_epoch{t_ckpt}.pt"),
            os.path.join(checkpoint_dir, f"model_f_{args.dataset}_{args.model_type}_epoch{t_ckpt}.pth")
        ]
        
        ckpt_filename = None
        for pattern in checkpoint_patterns:
            if os.path.exists(pattern):
                ckpt_filename = pattern
                break
                
        if not ckpt_filename:
            logger.error(f"Checkpoint file not found for epoch {t_ckpt}")
            logger.error(f"Searched for patterns: {checkpoint_patterns}")
            continue
        
        # Initialize model with same architecture
        try:
            # Get hidden_dim and num_layers from stage1 info if available
            hidden_dim = 32  # Default
            num_layers = 3   # Default
            gat_heads = 4    # Default
            
            # Look for run_info file to get exact parameters
            run_info_pattern = f"stage1_run_info_{args.dataset}_{args.model_type}_seed{args.stage1_seed}_*.json"
            import glob
            run_info_files = glob.glob(os.path.join(args.stage1_dir, run_info_pattern))
            
            if run_info_files:
                # Use the most recent run info file
                latest_run_info = sorted(run_info_files)[-1]
                with open(latest_run_info, 'r') as f:
                    run_info = json.load(f)
                hidden_dim = run_info.get('hidden_dim', hidden_dim)
                num_layers = run_info.get('num_layers', num_layers)
                gat_heads = run_info.get('gat_heads', gat_heads)
            
            model_f = get_model(
                model_type=args.model_type,
                num_features=data.x.size(1),
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                gat_heads=gat_heads
            )
            
            # Load checkpoint
            state_dict = torch.load(ckpt_filename, map_location='cpu')
            model_f.load_state_dict(state_dict)
            model_f.eval()
            
            logger.info(f"Successfully loaded model from {ckpt_filename}")
            
        except Exception as e:
            logger.error(f"Error initializing model from checkpoint: {e}")
            logger.error(traceback.format_exc())
            continue
        
        # Calculate NTK and alignments
        align_ntk_a, align_ntk_y = -1.0, -1.0
        
        try:
            # Calculate NTK with OOM handling
            Theta_t = calculate_ntk_with_oom_handling(model_f, data, logger, device)
            
            if Theta_t is not None:
                # Calculate alignments
                align_ntk_a = centered_kernel_alignment(Theta_t, A_dense.to(Theta_t.device))
                align_ntk_y = centered_kernel_alignment(Theta_t, Theta_star.to(Theta_t.device))
                
                logger.info(f"NTK-Graph Alignment (NTK-A) at epoch {t_ckpt}: {align_ntk_a:.4f}")
                logger.info(f"NTK-Label Alignment (NTK-Y) at epoch {t_ckpt}: {align_ntk_y:.4f}")
            else:
                logger.warning(f"Failed to calculate NTK for epoch {t_ckpt}")
                
        except Exception as e:
            logger.error(f"Error calculating alignments for epoch {t_ckpt}: {e}")
            logger.error(traceback.format_exc())
        
        # Store results
        ntk_results.append({
            'epoch': t_ckpt,
            'align_ntk_a': align_ntk_a,
            'align_ntk_y': align_ntk_y,
            'align_a_y_const': align_a_y_const,
            'memorization_rate': row['memorization_rate']
        })
        
        # Clean up to free memory
        del model_f
        if 'Theta_t' in locals():
            del Theta_t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Process and save results
    if ntk_results:
        # Convert to DataFrame
        ntk_df = pd.DataFrame(ntk_results)
        
        # Save to CSV
        csv_path = os.path.join(log_dir, f'ntk_results_{args.dataset}_{args.model_type}.csv')
        ntk_df.to_csv(csv_path, index=False)
        logger.info(f"Saved NTK results to {csv_path}")
        
        # Create visualizations
        try:
            # Plot temporal dynamics
            plot_temporal_dynamics(ntk_df, args.dataset, args.model_type, log_dir, logger)
            
            # Plot correlations
            plot_correlation(ntk_df, log_dir, logger)
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("No NTK results were generated")
    
    logger.info("Stage 2 analysis complete!")

if __name__ == '__main__':
    main()
