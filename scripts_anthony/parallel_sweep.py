"""
Parallel Hyperparameter Sweep for Operator Inference ROM on HPC.

This module provides MPI-parallel grid search over regularization parameters
for learning Operator Inference (OpInf) reduced-order models. Designed for
TACC Frontera and similar HPC systems.

Usage:
    ibrun python -m scripts_anthony.parallel_sweep --config cluster
    
    # Or with mpirun:
    mpirun -np 56 python -m scripts_anthony.parallel_sweep --config cluster

Requirements:
    - mpi4py
    - numpy
    - Pre-computed POD basis and projected data (from Step 1)

References:
    Peherstorfer, B., & Willcox, K. (2016). Data-driven operator inference
    for nonintrusive projection-based model reduction.
"""

import argparse
import time
import numpy as np
from itertools import product
from mpi4py import MPI

from opinf_for_hw.utils import get_x_sq
from opinf_for_hw.utils.opinf_utils import (
    bprint,
    solve_opinf_difference_model,
)


def evaluate_single_hyperparameter_set(
    params: dict,
    D_state: np.ndarray,
    D_state_2: np.ndarray,
    Y_state: np.ndarray,
    D_out_2: np.ndarray,
    D_out: np.ndarray,
    Y_Gamma: np.ndarray,
    X_state: np.ndarray,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    mean_Gamma_n_ref: float,
    std_Gamma_n_ref: float,
    mean_Gamma_c_ref: float,
    std_Gamma_c_ref: float,
    r: int,
    s: int,
    n_steps: int,
    training_end: int,
) -> list:
    """
    Evaluate a single state regularization parameter set with all output combinations.
    
    Parameters
    ----------
    params : dict
        Dictionary with keys 'alpha_state_lin', 'alpha_state_quad',
        'gamma_reg_lin', 'gamma_reg_quad'.
    D_state, D_state_2, Y_state : np.ndarray
        Pre-computed state learning matrices.
    D_out_2, D_out, Y_Gamma : np.ndarray
        Pre-computed output learning matrices.
    X_state : np.ndarray
        State training data for initial condition.
    mean_Xhat, scaling_Xhat : float
        Normalization parameters.
    mean_Gamma_n_ref, std_Gamma_n_ref : float
        Reference statistics for Gamma_n.
    mean_Gamma_c_ref, std_Gamma_c_ref : float
        Reference statistics for Gamma_c.
    r : int
        Number of POD modes.
    s : int
        Size of quadratic term (r*(r+1)/2).
    n_steps : int
        Number of time steps for integration.
    training_end : int
        Index marking end of training region.
    
    Returns
    -------
    list
        List of result dictionaries, one per output regularization combination.
    """
    alpha_state_lin = params['alpha_state_lin']
    alpha_state_quad = params['alpha_state_quad']
    gamma_reg_lin_arr = params['gamma_reg_lin']
    gamma_reg_quad_arr = params['gamma_reg_quad']
    
    d_state = r + s
    d_out = r + s + 1
    
    results = []
    
    # Solve state operator learning problem
    regg = np.zeros(d_state)
    regg[:r] = alpha_state_lin
    regg[r:r + s] = alpha_state_quad
    regularizer = np.diag(regg)
    D_state_reg = D_state_2 + regularizer
    
    O = np.linalg.solve(D_state_reg, np.dot(D_state.T, Y_state)).T
    A = O[:, :r]
    F = O[:, r:r + s]
    
    # Integrate learned model
    f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))
    u0 = X_state[0, :]
    is_nan, Xhat_pred = solve_opinf_difference_model(u0, n_steps, f)
    
    if is_nan:
        # Return empty results for NaN case
        for alpha_out_lin in gamma_reg_lin_arr:
            for alpha_out_quad in gamma_reg_quad_arr:
                results.append({
                    'is_nan': True,
                    'alpha_state_lin': alpha_state_lin,
                    'alpha_state_quad': alpha_state_quad,
                    'alpha_out_lin': alpha_out_lin,
                    'alpha_out_quad': alpha_out_quad,
                })
        return results
    
    X_OpInf_full = Xhat_pred.T
    
    # Prepare for output operator learning
    Xhat_OpInf_scaled = (X_OpInf_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
    Xhat_2_OpInf = get_x_sq(Xhat_OpInf_scaled)
    
    # Sweep over output regularization
    for alpha_out_lin in gamma_reg_lin_arr:
        for alpha_out_quad in gamma_reg_quad_arr:
            # Solve output operator learning
            regg_out = np.zeros(d_out)
            regg_out[:r] = alpha_out_lin
            regg_out[r:r + s] = alpha_out_quad
            regg_out[r + s:] = alpha_out_lin
            regularizer_out = np.diag(regg_out)
            D_out_reg = D_out_2 + regularizer_out
            
            O_out = np.linalg.solve(D_out_reg, np.dot(D_out.T, Y_Gamma.T)).T
            
            C = O_out[:, :r]
            G = O_out[:, r:r + s]
            c = O_out[:, r + s]
            
            # Compute predictions
            Y_OpInf = (
                C @ Xhat_OpInf_scaled.T
                + G @ Xhat_2_OpInf.T
                + c[:, np.newaxis]
            )
            
            ts_Gamma_n = Y_OpInf[0, :]
            ts_Gamma_c = Y_OpInf[1, :]
            
            # Compute error metrics
            mean_Gamma_n_OpInf = np.mean(ts_Gamma_n[:training_end])
            std_Gamma_n_OpInf = np.std(ts_Gamma_n[:training_end], ddof=1)
            mean_Gamma_c_OpInf = np.mean(ts_Gamma_c[:training_end])
            std_Gamma_c_OpInf = np.std(ts_Gamma_c[:training_end], ddof=1)
            
            mean_err_Gamma_n = np.abs(mean_Gamma_n_ref - mean_Gamma_n_OpInf) / np.abs(mean_Gamma_n_ref)
            std_err_Gamma_n = np.abs(std_Gamma_n_ref - std_Gamma_n_OpInf) / std_Gamma_n_ref
            mean_err_Gamma_c = np.abs(mean_Gamma_c_ref - mean_Gamma_c_OpInf) / np.abs(mean_Gamma_c_ref)
            std_err_Gamma_c = np.abs(std_Gamma_c_ref - std_Gamma_c_OpInf) / std_Gamma_c_ref
            
            total_error = (mean_err_Gamma_n + std_err_Gamma_n +
                          mean_err_Gamma_c + std_err_Gamma_c)
            
            results.append({
                'is_nan': False,
                'A': A.copy(),
                'F': F.copy(),
                'C': C.copy(),
                'G': G.copy(),
                'c': c.copy(),
                'total_error': total_error,
                'mean_err_Gamma_n': mean_err_Gamma_n,
                'std_err_Gamma_n': std_err_Gamma_n,
                'mean_err_Gamma_c': mean_err_Gamma_c,
                'std_err_Gamma_c': std_err_Gamma_c,
                'alpha_state_lin': alpha_state_lin,
                'alpha_state_quad': alpha_state_quad,
                'alpha_out_lin': alpha_out_lin,
                'alpha_out_quad': alpha_out_quad,
            })
    
    return results


def parallel_hyperparameter_sweep(
    ridge_alf_lin_all: np.ndarray,
    ridge_alf_quad_all: np.ndarray,
    gamma_reg_lin: np.ndarray,
    gamma_reg_quad: np.ndarray,
    D_state: np.ndarray,
    D_state_2: np.ndarray,
    Y_state: np.ndarray,
    D_out_2: np.ndarray,
    D_out: np.ndarray,
    Y_Gamma: np.ndarray,
    X_state: np.ndarray,
    mean_Xhat: np.ndarray,
    scaling_Xhat: float,
    mean_Gamma_n_ref: float,
    std_Gamma_n_ref: float,
    mean_Gamma_c_ref: float,
    std_Gamma_c_ref: float,
    r: int,
    n_steps: int,
    training_end: int,
    comm: MPI.Comm = None,
) -> list:
    """
    Perform MPI-parallel hyperparameter sweep.
    
    Distributes state regularization parameter combinations across MPI ranks.
    Each rank evaluates all output regularization combinations for its assigned
    state parameters.
    
    Parameters
    ----------
    ridge_alf_lin_all, ridge_alf_quad_all : np.ndarray
        State regularization parameter arrays.
    gamma_reg_lin, gamma_reg_quad : np.ndarray
        Output regularization parameter arrays.
    D_state, D_state_2, Y_state : np.ndarray
        State learning matrices.
    D_out_2, D_out, Y_Gamma : np.ndarray
        Output learning matrices.
    X_state : np.ndarray
        State training data.
    mean_Xhat, scaling_Xhat : float
        Normalization parameters.
    mean_Gamma_n_ref, std_Gamma_n_ref, mean_Gamma_c_ref, std_Gamma_c_ref : float
        Reference statistics.
    r : int
        Number of POD modes.
    n_steps : int
        Integration steps.
    training_end : int
        Training region end index.
    comm : MPI.Comm, optional
        MPI communicator. Defaults to MPI.COMM_WORLD.
    
    Returns
    -------
    list
        On rank 0: list of all results from all ranks.
        On other ranks: empty list.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    s = int(r * (r + 1) / 2)
    
    # Generate all state parameter combinations
    state_param_combos = list(product(ridge_alf_lin_all, ridge_alf_quad_all))
    n_state_combos = len(state_param_combos)
    
    if rank == 0:
        bprint(f"Parallel sweep: {n_state_combos} state combos across {size} ranks")
        print(f"  State params: {len(ridge_alf_lin_all)} x {len(ridge_alf_quad_all)}")
        print(f"  Output params: {len(gamma_reg_lin)} x {len(gamma_reg_quad)}")
        n_total = n_state_combos * len(gamma_reg_lin) * len(gamma_reg_quad)
        print(f"  Total combinations: {n_total}")
    
    # Distribute work across ranks
    combos_per_rank = n_state_combos // size
    remainder = n_state_combos % size
    
    if rank < remainder:
        start_idx = rank * (combos_per_rank + 1)
        end_idx = start_idx + combos_per_rank + 1
    else:
        start_idx = rank * combos_per_rank + remainder
        end_idx = start_idx + combos_per_rank
    
    my_combos = state_param_combos[start_idx:end_idx]
    
    if rank == 0:
        print(f"  Rank {rank}: processing {len(my_combos)} state combos")
    
    # Evaluate assigned combinations
    local_results = []
    local_start_time = time.time()
    
    for i, (alpha_lin, alpha_quad) in enumerate(my_combos):
        params = {
            'alpha_state_lin': alpha_lin,
            'alpha_state_quad': alpha_quad,
            'gamma_reg_lin': gamma_reg_lin,
            'gamma_reg_quad': gamma_reg_quad,
        }
        
        results = evaluate_single_hyperparameter_set(
            params=params,
            D_state=D_state,
            D_state_2=D_state_2,
            Y_state=Y_state,
            D_out_2=D_out_2,
            D_out=D_out,
            Y_Gamma=Y_Gamma,
            X_state=X_state,
            mean_Xhat=mean_Xhat,
            scaling_Xhat=scaling_Xhat,
            mean_Gamma_n_ref=mean_Gamma_n_ref,
            std_Gamma_n_ref=std_Gamma_n_ref,
            mean_Gamma_c_ref=mean_Gamma_c_ref,
            std_Gamma_c_ref=std_Gamma_c_ref,
            r=r,
            s=s,
            n_steps=n_steps,
            training_end=training_end,
        )
        
        local_results.extend(results)
        
        # Progress reporting (rank 0 only)
        if rank == 0 and (i + 1) % max(1, len(my_combos) // 10) == 0:
            elapsed = time.time() - local_start_time
            print(f"    Rank 0: {i+1}/{len(my_combos)} state combos ({elapsed:.1f}s)")
    
    local_elapsed = time.time() - local_start_time
    
    # Gather results to rank 0
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Flatten results
        combined_results = []
        for rank_results in all_results:
            combined_results.extend(rank_results)
        
        # Count statistics
        n_nan = sum(1 for r in combined_results if r.get('is_nan', False))
        n_valid = len(combined_results) - n_nan
        
        bprint("Parallel sweep complete")
        print(f"  Total models evaluated: {len(combined_results)}")
        print(f"  Valid models: {n_valid}")
        print(f"  NaN models: {n_nan}")
        
        return combined_results
    else:
        return []


def select_best_models(
    results: list,
    method: str = "top_k",
    num_top_models: int = 20,
    threshold_mean: float = 0.05,
    threshold_std: float = 0.30,
) -> list:
    """
    Select best models from sweep results.
    
    Parameters
    ----------
    results : list
        List of result dictionaries from parallel sweep.
    method : str
        Selection method: "top_k" or "threshold".
    num_top_models : int
        Number of models to keep (top_k method).
    threshold_mean : float
        Maximum mean error (threshold method).
    threshold_std : float
        Maximum std error (threshold method).
    
    Returns
    -------
    list
        List of (score, model) tuples for selected models.
    """
    from opinf_for_hw.utils.opinf_utils import TopKModels, ThresholdModels
    
    # Filter out NaN results
    valid_results = [r for r in results if not r.get('is_nan', False)]
    
    if method == "top_k":
        collector = TopKModels(k=num_top_models)
        for model in valid_results:
            collector.add(score=model['total_error'], model=model)
    elif method == "threshold":
        collector = ThresholdModels(
            threshold_mean=threshold_mean,
            threshold_std=threshold_std
        )
        for model in valid_results:
            collector.add(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return collector.get_best()


def save_ensemble_models(
    best_models: list,
    output_path: str,
    r: int,
    method: str = "top_k",
    threshold_mean: float = None,
    threshold_std: float = None,
    num_top_models: int = None,
) -> str:
    """
    Save ensemble models to NPZ file.
    
    Parameters
    ----------
    best_models : list
        List of (score, model) tuples.
    output_path : str
        Directory to save file.
    r : int
        Number of POD modes (for filename).
    method : str
        Selection method used.
    threshold_mean, threshold_std : float
        Threshold parameters (if applicable).
    num_top_models : int
        Number of models (if applicable).
    
    Returns
    -------
    str
        Path to saved file.
    """
    if method == "top_k":
        filename = f"ensemble_models_r{r}_topk{len(best_models)}.npz"
    else:
        filename = f"ensemble_models_r{r}_thresh{len(best_models)}.npz"
    
    filepath = output_path + filename
    
    # Build ensemble data dictionary
    ensemble_data = {
        'num_models': len(best_models),
        'selection_method': method,
        'r': r
    }
    
    if method == "threshold":
        ensemble_data['threshold_mean_error'] = threshold_mean
        ensemble_data['threshold_std_error'] = threshold_std
    else:
        ensemble_data['num_top_models'] = num_top_models
    
    # Save each model
    for i, (score, model) in enumerate(best_models):
        prefix = f'model_{i}_'
        for key in ['A', 'F', 'C', 'G', 'c', 'alpha_state_lin', 'alpha_state_quad',
                    'alpha_out_lin', 'alpha_out_quad', 'total_error',
                    'mean_err_Gamma_n', 'std_err_Gamma_n',
                    'mean_err_Gamma_c', 'std_err_Gamma_c']:
            ensemble_data[prefix + key] = model[key]
    
    np.savez(filepath, **ensemble_data)
    
    return filepath


# =============================================================================
# MAIN ENTRY POINT FOR HPC
# =============================================================================

def main():
    """Main entry point for parallel hyperparameter sweep."""
    parser = argparse.ArgumentParser(
        description="Parallel OpInf hyperparameter sweep for HPC"
    )
    parser.add_argument(
        "--config", type=str, default="cluster",
        choices=["cluster", "local"],
        help="Configuration to use"
    )
    parser.add_argument(
        "--method", type=str, default="threshold",
        choices=["top_k", "threshold"],
        help="Model selection method"
    )
    parser.add_argument(
        "--num-top-models", type=int, default=20,
        help="Number of top models (top_k method)"
    )
    parser.add_argument(
        "--threshold-mean", type=float, default=0.05,
        help="Mean error threshold (threshold method)"
    )
    parser.add_argument(
        "--threshold-std", type=float, default=0.30,
        help="Std error threshold (threshold method)"
    )
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Load configuration
    if args.config == "cluster":
        from config.cluster import (
            output_path, ridge_alf_lin_all, ridge_alf_quad_all,
            gamma_reg_lin, gamma_reg_quad, n_steps, training_files
        )
        from config.HW import r, training_end
    else:
        from config.local import (
            output_path, ridge_alf_lin_all, ridge_alf_quad_all,
            gamma_reg_lin, gamma_reg_quad, n_steps, training_files
        )
        from config.HW import r, training_end
    
    if rank == 0:
        bprint("=" * 60)
        bprint("PARALLEL OPERATOR INFERENCE HYPERPARAMETER SWEEP")
        bprint("=" * 60)
        print(f"Configuration: {args.config}")
        print(f"Selection method: {args.method}")
        print(f"Number of MPI ranks: {comm.Get_size()}")
    
    # =========================================================================
    # SHARED MEMORY DATA LOADING
    # =========================================================================
    # Use MPI shared memory to avoid duplicating large arrays on each rank.
    # Only rank 0 on each node loads data; other ranks share that memory.
    # =========================================================================
    
    # Create node-local communicator for shared memory
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()
    
    if rank == 0:
        bprint(f"\nMemory optimization: {node_size} ranks sharing memory per node")
        bprint("Loading pre-computed data (rank 0 per node)...")
    
    s = int(r * (r + 1) / 2)
    d_state = r + s
    d_out = r + s + 1
    
    # ---------- Step 1: Rank 0 loads data and determines shapes ----------
    if rank == 0:
        # Load projected training data
        Xhat_train_local = np.load(output_path + "X_hat_train_multi_IC.npy")
        if Xhat_train_local.shape[1] > r:
            Xhat_train_local = Xhat_train_local[:, :r]
        
        # Load boundaries
        boundaries_data = np.load(output_path + "data_boundaries.npz")
        train_boundaries_local = boundaries_data['train_boundaries']
        
        # Prepare state learning data
        n_train_traj = len(train_boundaries_local) - 1
        X_state_list = []
        Y_state_list = []
        
        for traj_idx in range(n_train_traj):
            start_idx = train_boundaries_local[traj_idx]
            end_idx = train_boundaries_local[traj_idx + 1]
            Xhat_traj = Xhat_train_local[start_idx:end_idx, :]
            X_state_list.append(Xhat_traj[:-1, :])
            Y_state_list.append(Xhat_traj[1:, :])
        
        X_state_local = np.vstack(X_state_list)
        Y_state_local = np.vstack(Y_state_list)
        
        X_state2 = get_x_sq(X_state_local)
        D_state_local = np.concatenate((X_state_local, X_state2), axis=1)
        D_state_2_local = D_state_local.T @ D_state_local
        
        # Prepare output learning data
        X_out = Xhat_train_local
        K = X_out.shape[0]
        E = np.ones((K, 1))
        
        mean_Xhat_local = np.mean(X_out, axis=0)
        Xhat_out = X_out - mean_Xhat_local[np.newaxis, :]
        scaling_Xhat_local = np.maximum(np.abs(np.min(X_out)), np.abs(np.max(X_out)))
        Xhat_out /= scaling_Xhat_local
        Xhat_out2 = get_x_sq(Xhat_out)
        
        D_out_local = np.concatenate((Xhat_out, Xhat_out2, E), axis=1)
        D_out_2_local = D_out_local.T @ D_out_local
        
        # Load reference Gamma values
        from opinf_for_hw.utils.helpers import loader
        Gamma_n_list = []
        Gamma_c_list = []
        for file_path in training_files:
            fh = loader(file_path, ENGINE="h5netcdf")
            Gamma_n_list.append(fh["gamma_n"].data)
            Gamma_c_list.append(fh["gamma_c"].data)
        
        Gamma_n = np.concatenate(Gamma_n_list)
        Gamma_c = np.concatenate(Gamma_c_list)
        Y_Gamma_local = np.vstack((Gamma_n, Gamma_c))
        
        mean_Gamma_n_ref = np.mean(Gamma_n)
        std_Gamma_n_ref = np.std(Gamma_n, ddof=1)
        mean_Gamma_c_ref = np.mean(Gamma_c)
        std_Gamma_c_ref = np.std(Gamma_c, ddof=1)
        
        # Collect shapes for broadcasting
        shapes = {
            'X_state': X_state_local.shape,
            'Y_state': Y_state_local.shape,
            'D_state': D_state_local.shape,
            'D_state_2': D_state_2_local.shape,
            'D_out': D_out_local.shape,
            'D_out_2': D_out_2_local.shape,
            'Y_Gamma': Y_Gamma_local.shape,
            'mean_Xhat': mean_Xhat_local.shape,
        }
        scalars = {
            'scaling_Xhat': scaling_Xhat_local,
            'mean_Gamma_n_ref': mean_Gamma_n_ref,
            'std_Gamma_n_ref': std_Gamma_n_ref,
            'mean_Gamma_c_ref': mean_Gamma_c_ref,
            'std_Gamma_c_ref': std_Gamma_c_ref,
        }
        
        bprint("Data loaded successfully")
        print(f"  X_state shape: {X_state_local.shape}")
        print(f"  D_out shape: {D_out_local.shape}")
    else:
        shapes = None
        scalars = None
    
    # ---------- Step 2: Broadcast shapes and scalars to all ranks ----------
    shapes = comm.bcast(shapes, root=0)
    scalars = comm.bcast(scalars, root=0)
    
    scaling_Xhat = scalars['scaling_Xhat']
    mean_Gamma_n_ref = scalars['mean_Gamma_n_ref']
    std_Gamma_n_ref = scalars['std_Gamma_n_ref']
    mean_Gamma_c_ref = scalars['mean_Gamma_c_ref']
    std_Gamma_c_ref = scalars['std_Gamma_c_ref']
    
    # ---------- Step 3: Create shared memory windows on each node ----------
    def create_shared_array(node_comm, shape, dtype=np.float64):
        """Create a shared memory array accessible by all ranks on a node."""
        size = int(np.prod(shape))
        itemsize = np.dtype(dtype).itemsize
        
        if node_comm.Get_rank() == 0:
            nbytes = size * itemsize
        else:
            nbytes = 0
        
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
        buf, itemsize = win.Shared_query(0)
        arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
        return arr, win
    
    # Allocate shared arrays
    X_state, win_X_state = create_shared_array(node_comm, shapes['X_state'])
    Y_state, win_Y_state = create_shared_array(node_comm, shapes['Y_state'])
    D_state, win_D_state = create_shared_array(node_comm, shapes['D_state'])
    D_state_2, win_D_state_2 = create_shared_array(node_comm, shapes['D_state_2'])
    D_out, win_D_out = create_shared_array(node_comm, shapes['D_out'])
    D_out_2, win_D_out_2 = create_shared_array(node_comm, shapes['D_out_2'])
    Y_Gamma, win_Y_Gamma = create_shared_array(node_comm, shapes['Y_Gamma'])
    mean_Xhat, win_mean_Xhat = create_shared_array(node_comm, shapes['mean_Xhat'])
    
    # ---------- Step 4: Rank 0 on each node fills shared memory ----------
    # Global rank 0 broadcasts data to node-rank-0 on all nodes
    if node_rank == 0:
        if rank == 0:
            # Rank 0 has the data, copy to shared memory
            X_state[:] = X_state_local
            Y_state[:] = Y_state_local
            D_state[:] = D_state_local
            D_state_2[:] = D_state_2_local
            D_out[:] = D_out_local
            D_out_2[:] = D_out_2_local
            Y_Gamma[:] = Y_Gamma_local
            mean_Xhat[:] = mean_Xhat_local
        else:
            # Other node-rank-0s receive via broadcast
            pass
    
    # Broadcast arrays to node-rank-0 on each node
    # Create communicator of just node-rank-0s
    node_root_comm = comm.Split(color=0 if node_rank == 0 else MPI.UNDEFINED, key=rank)
    
    if node_rank == 0:
        # Broadcast from global rank 0 to all node-rank-0s
        node_root_comm.Bcast(X_state, root=0)
        node_root_comm.Bcast(Y_state, root=0)
        node_root_comm.Bcast(D_state, root=0)
        node_root_comm.Bcast(D_state_2, root=0)
        node_root_comm.Bcast(D_out, root=0)
        node_root_comm.Bcast(D_out_2, root=0)
        node_root_comm.Bcast(Y_Gamma, root=0)
        node_root_comm.Bcast(mean_Xhat, root=0)
    
    # Synchronize within node so all ranks see the data
    node_comm.Barrier()
    
    if rank == 0:
        bprint("Shared memory arrays populated on all nodes")
    
    # Synchronize before sweep
    comm.Barrier()
    
    # Run parallel sweep
    if rank == 0:
        bprint("\nStarting parallel hyperparameter sweep...")
    
    sweep_start = time.time()
    
    results = parallel_hyperparameter_sweep(
        ridge_alf_lin_all=ridge_alf_lin_all,
        ridge_alf_quad_all=ridge_alf_quad_all,
        gamma_reg_lin=gamma_reg_lin,
        gamma_reg_quad=gamma_reg_quad,
        D_state=D_state,
        D_state_2=D_state_2,
        Y_state=Y_state,
        D_out_2=D_out_2,
        D_out=D_out,
        Y_Gamma=Y_Gamma,
        X_state=X_state,
        mean_Xhat=mean_Xhat,
        scaling_Xhat=scaling_Xhat,
        mean_Gamma_n_ref=mean_Gamma_n_ref,
        std_Gamma_n_ref=std_Gamma_n_ref,
        mean_Gamma_c_ref=mean_Gamma_c_ref,
        std_Gamma_c_ref=std_Gamma_c_ref,
        r=r,
        n_steps=n_steps,
        training_end=training_end,
        comm=comm,
    )
    
    sweep_elapsed = time.time() - sweep_start
    
    # Rank 0 handles model selection and saving
    if rank == 0:
        bprint(f"\nSweep completed in {sweep_elapsed:.1f}s ({sweep_elapsed/60:.1f} min)")
        
        # Select best models
        best_models = select_best_models(
            results=results,
            method=args.method,
            num_top_models=args.num_top_models,
            threshold_mean=args.threshold_mean,
            threshold_std=args.threshold_std,
        )
        
        bprint(f"\nSelected {len(best_models)} models using '{args.method}' method")
        
        if len(best_models) > 0:
            print(f"  Best total error: {best_models[0][0]:.6e}")
            print(f"  Worst selected: {best_models[-1][0]:.6e}")
            
            # Save models
            filepath = save_ensemble_models(
                best_models=best_models,
                output_path=output_path,
                r=r,
                method=args.method,
                threshold_mean=args.threshold_mean,
                threshold_std=args.threshold_std,
                num_top_models=args.num_top_models,
            )
            bprint(f"\nSaved ensemble to: {filepath}")
        else:
            bprint("WARNING: No models met selection criteria!")
        
        bprint("=" * 60)
        bprint("SWEEP COMPLETE")
        bprint("=" * 60)
    
    # Cleanup shared memory windows
    comm.Barrier()
    win_X_state.Free()
    win_Y_state.Free()
    win_D_state.Free()
    win_D_state_2.Free()
    win_D_out.Free()
    win_D_out_2.Free()
    win_Y_Gamma.Free()
    win_mean_Xhat.Free()


if __name__ == "__main__":
    main()
