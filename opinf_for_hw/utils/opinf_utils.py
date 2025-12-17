"""
Utility functions for Operator Inference (OpInf) reduced-order modeling.

This module provides common helper functions for:
- Console formatting
- File I/O for memory-mapped arrays
- Time discretization utilities
- Model integration
- Model selection (Top-K and Threshold-based)

References:
    Peherstorfer, B., & Willcox, K. (2016). Data-driven operator inference
    for nonintrusive projection-based model reduction.
"""

import os
import heapq
import numpy as np
import h5py


# =============================================================================
# CONSOLE FORMATTING
# =============================================================================

def bprint(msg: str) -> None:
    """Print bold text to console using ANSI escape codes."""
    print("\033[1m" + msg + "\033[0m")


# =============================================================================
# MEMORY-MAPPED FILE UTILITIES
# =============================================================================

def get_memmap_path(output_path: str, name: str) -> str:
    """
    Get full path for a memory-mapped file.
    
    Parameters
    ----------
    output_path : str
        Directory where memmap files are stored.
    name : str
        Base name for the memmap file.
    
    Returns
    -------
    str
        Full path to memmap file.
    """
    return os.path.join(output_path, f"memmap_{name}.dat")


def cleanup_memmap(output_path: str, name: str) -> None:
    """
    Remove a memory-mapped file if it exists.
    
    Parameters
    ----------
    output_path : str
        Directory where memmap files are stored.
    name : str
        Base name for the memmap file.
    """
    path = get_memmap_path(output_path, name)
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# TIME DISCRETIZATION UTILITIES
# =============================================================================

def get_dt_from_file(file_path: str, default: float = 0.025) -> float:
    """
    Extract time step (dt) from HDF5 file attributes.
    
    Searches common attribute locations in the HDF5 file structure.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    default : float, optional
        Default dt value if not found in file. Default is 0.025.
    
    Returns
    -------
    float
        Time step value.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Check root attributes
            if 'dt' in f.attrs:
                return float(f.attrs['dt'])
            if 'dt' in f:
                return float(f['dt'][()])
            
            # Check common group locations
            for group_name in ['params', 'metadata', 'parameters']:
                if group_name in f:
                    grp = f[group_name]
                    if 'dt' in grp.attrs:
                        return float(grp.attrs['dt'])
                    if 'dt' in grp:
                        return float(grp['dt'][()])
    except Exception as e:
        print(f"  Warning: Could not read dt from {file_path}: {e}")
    
    return default


def compute_truncation_snapshots(
    file_path: str,
    truncate_snapshots: int = None,
    truncate_time: float = None,
    default_dt: float = 0.025
) -> int:
    """
    Compute number of snapshots to keep based on truncation settings.
    
    Parameters
    ----------
    file_path : str
        Path to data file (used to extract dt).
    truncate_snapshots : int, optional
        Direct number of snapshots to keep.
    truncate_time : float, optional
        Simulation time to keep (converted to snapshots using dt).
    default_dt : float, optional
        Default time step if not found in file.
    
    Returns
    -------
    int or None
        Number of snapshots to keep, or None if no truncation.
    
    Notes
    -----
    If both truncate_snapshots and truncate_time are provided,
    truncate_snapshots takes priority.
    """
    if truncate_snapshots is not None:
        return truncate_snapshots
    elif truncate_time is not None:
        dt = get_dt_from_file(file_path, default_dt)
        n_snaps = int(truncate_time / dt)
        print(f"    Using dt={dt:.4f} -> {n_snaps} snapshots for t={truncate_time}")
        return n_snaps
    return None


# =============================================================================
# MODEL INTEGRATION
# =============================================================================

def solve_opinf_difference_model(s0: np.ndarray, n_steps: int, f: callable):
    """
    Integrate a discrete-time dynamical system forward.
    
    Solves the difference equation: s_{k+1} = f(s_k)
    
    Parameters
    ----------
    s0 : np.ndarray
        Initial state vector of shape (r,).
    n_steps : int
        Number of time steps to integrate.
    f : callable
        State transition function f: R^r -> R^r.
    
    Returns
    -------
    is_nan : bool
        True if NaN values were encountered during integration.
    s : np.ndarray
        State trajectory of shape (r, n_steps).
    
    Examples
    --------
    >>> A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    >>> f = lambda x: A @ x
    >>> s0 = np.array([1.0, 0.0])
    >>> is_nan, trajectory = solve_opinf_difference_model(s0, 100, f)
    """
    r = np.size(s0)
    s = np.zeros((r, n_steps))
    is_nan = False

    s[:, 0] = s0
    for i in range(n_steps - 1):
        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):
            print(f"NaN encountered at iteration {i + 1}")
            is_nan = True
            break

    return is_nan, s


# =============================================================================
# MODEL SELECTION CLASSES
# =============================================================================

class TopKModels:
    """
    Maintains a collection of the k best models based on score.
    
    Uses a min-heap for efficient O(log k) insertion while maintaining
    only the k best models (lowest scores).
    
    Parameters
    ----------
    k : int
        Number of top models to retain.
    
    Attributes
    ----------
    heap : list
        Min-heap of (score, model) tuples.
    
    Examples
    --------
    >>> collector = TopKModels(k=5)
    >>> collector.add(score=0.1, model={'name': 'model_1'})
    >>> collector.add(score=0.05, model={'name': 'model_2'})
    >>> best = collector.get_best()
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        self.heap = []

    def add(self, score: float, model: dict) -> None:
        """
        Add a model to the collection if it's among the k best.
        
        Parameters
        ----------
        score : float
            Model score (lower is better).
        model : dict
            Model data dictionary.
        """
        entry = (score, model)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        else:
            if score < self.heap[0][0]:
                heapq.heapreplace(self.heap, entry)

    def get_best(self) -> list:
        """
        Return models sorted from best to worst.
        
        Returns
        -------
        list
            List of (score, model) tuples sorted by ascending score.
        """
        return sorted(self.heap, reverse=False)


class ThresholdModels:
    """
    Collects all models meeting specified error thresholds.
    
    A model is accepted if ALL of the following criteria are met:
        - mean_err_Gamma_n < threshold_mean
        - std_err_Gamma_n < threshold_std
        - mean_err_Gamma_c < threshold_mean
        - std_err_Gamma_c < threshold_std
    
    Parameters
    ----------
    threshold_mean : float
        Maximum allowed relative error in mean prediction.
    threshold_std : float
        Maximum allowed relative error in standard deviation prediction.
    
    Examples
    --------
    >>> collector = ThresholdModels(threshold_mean=0.05, threshold_std=0.30)
    >>> model = {
    ...     'mean_err_Gamma_n': 0.03,
    ...     'std_err_Gamma_n': 0.20,
    ...     'mean_err_Gamma_c': 0.04,
    ...     'std_err_Gamma_c': 0.25,
    ...     'total_error': 0.52
    ... }
    >>> accepted = collector.add(model)  # Returns True
    """
    
    def __init__(self, threshold_mean: float = 0.05, threshold_std: float = 0.30):
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.models = []

    def add(self, model: dict) -> bool:
        """
        Add model if it meets all threshold criteria.
        
        Parameters
        ----------
        model : dict
            Model dictionary containing error metrics.
        
        Returns
        -------
        bool
            True if model was accepted, False otherwise.
        """
        meets_criteria = (
            model['mean_err_Gamma_n'] < self.threshold_mean and
            model['std_err_Gamma_n'] < self.threshold_std and
            model['mean_err_Gamma_c'] < self.threshold_mean and
            model['std_err_Gamma_c'] < self.threshold_std
        )
        if meets_criteria:
            self.models.append((model['total_error'], model))
            return True
        return False

    def get_best(self) -> list:
        """
        Return models sorted from best to worst by total error.
        
        Returns
        -------
        list
            List of (score, model) tuples sorted by ascending total_error.
        """
        return sorted(self.models, key=lambda x: x[0])
