import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev
import gc
import traceback
import sys
import time # For timed progress updates

# empirical_ntk function remains unchanged from your previous version
@torch.no_grad()
def empirical_ntk(model, X, edge_index=None, type='nknk'):
    model.eval()
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    def f(functional_params, x_input):
        if edge_index is not None: call_args = (x_input, edge_index)
        else: call_args = (x_input,)
        return functional_call(model, functional_params, args=call_args)
    jacobian_dict = jacrev(f, argnums=0)(params, X)
    jacobian_flat_list = []
    for p_name in params:
         if p_name in jacobian_dict:
              jac_tensor = jacobian_dict[p_name]
              if jac_tensor.dim() > 2 + len(params[p_name].shape):
                  num_extra_dims = jac_tensor.dim() - 2 - len(params[p_name].shape)
                  flattened_jac = jac_tensor.flatten(start_dim=2+num_extra_dims)
                  if num_extra_dims > 0:
                       flattened_jac = flattened_jac.flatten(start_dim=1, end_dim=1+num_extra_dims)
              else:
                  flattened_jac = jac_tensor.flatten(start_dim=1)
              jacobian_flat_list.append(flattened_jac)
    if not jacobian_flat_list: raise ValueError("Jacobian computation resulted in empty list.")
    jacobian = torch.cat(jacobian_flat_list, dim=-1)
    if type == 'nknk': ntk_mat = torch.einsum('nkp,NKp->nkNK', jacobian, jacobian)
    elif type == 'nn': ntk_mat = torch.einsum('...p,...p->nN', jacobian, jacobian)
    else: raise ValueError(f"Unknown NTK type: {type}")
    return ntk_mat

@torch.no_grad()
def empirical_ntk_blocked(model, X, edge_index=None, block_size=128, sample_size=None, progress_interval_seconds=60):
    """
    Computes the empirical NTK (type 'nn') for GNNs in blocks, optionally on
    a subsample of nodes. Prints progress updates at intervals.

    Args:
        model: The neural network model.
        X: Input features (N_orig x F), should be on CPU initially.
        edge_index: Edge index for GNNs, should be on CPU initially.
        block_size: Size of blocks to process.
        sample_size (int, optional): If set, compute NTK on a random sample.
        progress_interval_seconds (int): How often to print progress (in seconds).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The NTK matrix on CPU.
            - The indices of nodes used (relative to original N_orig), on CPU.
              Returns (None, None) if calculation fails.
    """
    model.eval()
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    N_orig = X.shape[0]

    # --- Subsampling Logic ---
    node_indices_cpu = None
    if sample_size is not None:
        if sample_size <= 0:
             print("Info: sample_size invalid. Computing full NTK.", flush=True)
             N = N_orig; node_indices_cpu = torch.arange(N_orig, device='cpu')
        elif sample_size < N_orig:
            print(f"Info: Subsampling to {sample_size} / {N_orig} nodes.", flush=True)
            sampled_indices_cpu = torch.randperm(N_orig)[:sample_size]
            node_indices_cpu = torch.sort(sampled_indices_cpu).values
            N = sample_size
        else:
            print(f"Info: sample_size >= N_orig. Computing full NTK.", flush=True)
            N = N_orig; node_indices_cpu = torch.arange(N_orig, device='cpu')
    else:
        print("Info: Computing full NTK.", flush=True)
        N = N_orig; node_indices_cpu = torch.arange(N_orig, device='cpu')

    # --- Device Selection & Data Moving ---
    primary_device = next(model.parameters()).device
    print(f"Info: NTK attempt on primary_device={primary_device}, block_size={block_size}, N_eff={N}", flush=True)
    compute_device = primary_device
    X_device, edge_index_device = None, None
    try:
        X_device = X.to(compute_device)
        edge_index_device = edge_index.to(compute_device) if edge_index is not None else None
    except RuntimeError:
        print(f"Warning: OOM moving full data to {compute_device}. Falling back to CPU.", flush=True)
        compute_device = torch.device('cpu')
        X_device = X.cpu(); edge_index_device = edge_index.cpu() if edge_index is not None else None
        model.to(compute_device); params = {name: p.to(compute_device) for name, p in model.named_parameters() if p.requires_grad}
    gc.collect(); torch.cuda.empty_cache() if primary_device.type == 'cuda' else None

    # --- Define 'f' for functional_call ---
    def f(functional_params, current_block_indices_rel):
        original_node_indices = node_indices_cpu[current_block_indices_rel]
        original_node_indices_device = original_node_indices.to(compute_device) # Slicing indices
        if edge_index_device is not None:
            full_output = functional_call(model, functional_params, (X_device, edge_index_device))
        else:
            full_output = functional_call(model, functional_params, (X_device,))
        return full_output[original_node_indices_device]

    ntk_mat = torch.zeros((N, N), dtype=torch.float32, device='cpu')
    print(f"Info: Allocated CPU NTK matrix of size {N}x{N}", flush=True)
    params_on_compute_device = {k: v.to(compute_device) for k, v in params.items()}

    # --- Progress Tracking Initialization ---
    total_blocks = (N + block_size - 1) // block_size
    total_pairs_to_process = total_blocks * (total_blocks + 1) // 2
    processed_pairs = 0
    last_progress_print_time = time.time()
    start_time = time.time()
    print(f"Info: Starting NTK block computation. Total block pairs to process: {total_pairs_to_process}", flush=True)

    # --- Main Computation Loop ---
    try:
        for i in range(0, N, block_size):
            end_i = min(i + block_size, N)
            indices_i_rel = torch.arange(i, end_i, device='cpu')
            jac_i, jac_i_dict = None, None
            try:
                # print(f"  Debug: Computing Jac_i for block {i//block_size+1}/{total_blocks}", flush=True)
                def func_wrapper_i(p): return f(p, indices_i_rel)
                jac_i_dict = jacrev(func_wrapper_i, argnums=0)(params_on_compute_device)
                jac_i_list = [jac_i_dict[p_name].flatten(start_dim=1) for p_name in params if p_name in jac_i_dict]
                if not jac_i_list: raise ValueError(f"Jacobian i ({i}) empty list.")
                jac_i = torch.cat(jac_i_list, dim=-1).to(compute_device)
            except Exception as e:
                print(f"Error during Jacobian i ({i}) computation: {e}\n{traceback.format_exc()}", flush=True)
                raise
            finally:
                if jac_i_dict is not None: del jac_i_dict

            for j in range(i, N, block_size):
                end_j = min(j + block_size, N)
                indices_j_rel = torch.arange(j, end_j, device='cpu')
                jac_j, jac_j_dict = None, None
                if i == j: jac_j = jac_i
                else:
                    try:
                        # print(f"    Debug: Computing Jac_j for block {j//block_size+1}/{total_blocks}", flush=True)
                        def func_wrapper_j(p): return f(p, indices_j_rel)
                        jac_j_dict = jacrev(func_wrapper_j, argnums=0)(params_on_compute_device)
                        jac_j_list = [jac_j_dict[p_name].flatten(start_dim=1) for p_name in params if p_name in jac_j_dict]
                        if not jac_j_list: raise ValueError(f"Jacobian j ({j}) empty list.")
                        jac_j = torch.cat(jac_j_list, dim=-1).to(compute_device)
                    except Exception as e:
                        print(f"Error during Jacobian j ({j}) computation: {e}\n{traceback.format_exc()}", flush=True)
                        raise
                    finally:
                         if jac_j_dict is not None: del jac_j_dict

                ntk_block = None
                try:
                    # print(f"      Debug: Computing NTK block ({i//block_size+1},{j//block_size+1})", flush=True)
                    ntk_block = torch.einsum('ip,jp->ij', jac_i.float(), jac_j.float())
                except Exception as e:
                    print(f"Error during NTK block computation ({i},{j}): {e}\n{traceback.format_exc()}", flush=True)
                    raise
                finally:
                    if i != j and jac_j is not None: del jac_j

                ntk_mat[i:end_i, j:end_j] = ntk_block.cpu()
                if i != j: ntk_mat[j:end_j, i:end_i] = ntk_block.T.cpu()
                del ntk_block
                processed_pairs += 1

                # --- Timed Progress Update ---
                current_time = time.time()
                if current_time - last_progress_print_time >= progress_interval_seconds or processed_pairs == total_pairs_to_process:
                    elapsed_time_total = current_time - start_time
                    progress_percent = (processed_pairs / total_pairs_to_process) * 100
                    time_per_pair = elapsed_time_total / processed_pairs if processed_pairs > 0 else 0
                    estimated_remaining_time = (total_pairs_to_process - processed_pairs) * time_per_pair if processed_pairs > 0 else float('inf')
                    
                    print(f"Progress: {processed_pairs}/{total_pairs_to_process} pairs ({progress_percent:.1f}%) completed. "
                          f"Elapsed: {elapsed_time_total:.1f}s. "
                          f"Est. Remaining: {estimated_remaining_time:.1f}s.", flush=True)
                    last_progress_print_time = current_time
                    # Optionally log memory usage here if needed
                    # if compute_device.type == 'cuda':
                    #     print(f"    CUDA Mem: Allocated={torch.cuda.memory_allocated(compute_device)/1e9:.2f}GB, "
                    #           f"Reserved={torch.cuda.memory_reserved(compute_device)/1e9:.2f}GB", flush=True)

            # Cleanup jac_i outside inner loop
            del jac_i
            gc.collect()
            if compute_device.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
         print(f"\nError during NTK block computation loop: {e}", flush=True)
         print(traceback.format_exc(), flush=True)
         del ntk_mat, params_on_compute_device # X_device, edge_index_device cleaned later
         gc.collect()
         if compute_device.type == 'cuda': torch.cuda.empty_cache()
         return None, None # Return None on error

    # --- Final Cleanup ---
    if 'X_device' in locals() and X_device is not None: del X_device
    if 'edge_index_device' in locals() and edge_index_device is not None: del edge_index_device
    if 'params_on_compute_device' in locals(): del params_on_compute_device
    gc.collect()
    if primary_device.type == 'cuda': torch.cuda.empty_cache()

    total_computation_time = time.time() - start_time
    print(f"Info: NTK computation finished. Total time: {total_computation_time:.2f}s", flush=True)
    return ntk_mat, node_indices_cpu


@torch.no_grad()
def centered_kernel_alignment(K1, K2):
    # --- (Implementation remains the same) ---
    K1 = K1.float(); K2 = K2.float(); n = K1.shape[0]
    if n == 0: return 0.0
    K1_mean_row = K1.mean(dim=1, keepdim=True); K1_mean_col = K1.mean(dim=0, keepdim=True); K1_mean_all = K1.mean()
    centered_K1 = K1 - K1_mean_row - K1_mean_col + K1_mean_all
    K2_mean_row = K2.mean(dim=1, keepdim=True); K2_mean_col = K2.mean(dim=0, keepdim=True); K2_mean_all = K2.mean()
    centered_K2 = K2 - K2_mean_row - K2_mean_col + K2_mean_all
    cka_numerator = (centered_K1 * centered_K2).sum()
    norm_K1_c = torch.norm(centered_K1, p='fro'); norm_K2_c = torch.norm(centered_K2, p='fro')
    cka_denominator = norm_K1_c * norm_K2_c
    if cka_denominator < 1e-9: return 0.0 if abs(cka_numerator) < 1e-9 else torch.sign(cka_numerator).item()
    alignment = cka_numerator / cka_denominator
    alignment = torch.clamp(alignment, min=-1.0, max=1.0)
    return alignment.item()