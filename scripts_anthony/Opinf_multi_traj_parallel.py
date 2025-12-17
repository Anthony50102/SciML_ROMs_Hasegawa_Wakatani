# %% [markdown]
# # End-to-End Opinf for multiple initial conditions
# ## Install notes with current startup script
# - `pip install -e . --no-deps`  # Install without dependencies first
# - `pip install -e .`            # Then install with pinned dependencies
# ## What this notebook can do
# - Compute a POD basis for 1+ intial conditions
# - Find the operator(s) for $\gamma_n$, $\gamma_c$, and state
# - Use learned operator(s) and POD to compute predictions across unseen data
# 
# ## Configuration Notes
# 
# ### Memory-mapped files
# Files created in `output_path` during execution:
# - `memmap_Q_train.dat` - Full training snapshots (step_1 only)
# - `memmap_Q_test.dat` - Full test snapshots (step_1 only)  
# 
# ### Pipeline control
# | Setting | Description |
# |---------|-------------|
# | `step_1=True, step_2=True` | Full pipeline: compute POD, train ROM, make predictions |
# | `step_1=False, step_2=True` | Load existing POD, train new ROM |
# | `step_1=False, step_2=False` | Load existing POD and ROM, make predictions only |
# 
# ### Data truncation
# Limit snapshots per trajectory for quick tests or memory constraints:
# ```python
# truncate_data = True
# truncate_snapshots = 1000      # Option 1: Keep N snapshots
# # OR
# truncate_time = 25.0           # Option 2: Keep T time units (uses dt from file)
# ```
# 
# ### Ensemble model selection
# Two methods for selecting models to include in the ensemble:
# 
# **Method 1: Top-K Selection**
# ```python
# model_selection_method = "top_k"
# num_top_models = 20            # Keep the 20 best models
# ```
# 
# **Method 2: Threshold Selection**
# ```python
# model_selection_method = "threshold"
# threshold_mean_error = 0.05    # Accept if mean relative error < 5%
# threshold_std_error = 0.30     # Accept if std relative error < 30%
# ```
# 
# ### Cleanup
# To free disk space after completion:
# ```python
# for name in ["Q_train", "Q_test"]:
#     cleanup_memmap(name)
# ```

# %%
%matplotlib inline
from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *
from opinf_for_hw.utils.helpers import loader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from IPython import display
import xarray as xr
import time
import gc
import os

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
cluster = True
show_animations = False
show_plots = True
ENGINE = "h5netcdf"
plt.rcParams['animation.embed_limit'] = 250
plt.rcParams["image.cmap"] = "bwr"  # Other options: seismic

# =============================================================================
# PIPELINE CONTROL
# =============================================================================
step_1 = False  # Compute POD basis from training data
step_2 = True  # Train ROM and perform regularization sweep

# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================
use_chunked_projection = False  # Set True to project in chunks (lower peak memory)

# =============================================================================
# DATA TRUNCATION CONFIGURATION
# =============================================================================
# Set truncate_data = True to limit the number of snapshots used.
# Specify EITHER truncate_snapshots OR truncate_time (not both).
# If both are set, truncate_snapshots takes priority.
# =============================================================================
truncate_data = True           # Enable/disable truncation
truncate_snapshots = None       # Number of snapshots to keep (e.g., 1000)
truncate_time = 200            # Simulation time to keep (e.g., 25.0 time units)
default_dt = 0.025              # Default dt if not found in file attributes

# =============================================================================
# ENSEMBLE MODEL SELECTION CONFIGURATION
# =============================================================================
# Two methods for selecting models to include in the ensemble:
#
# Method 1: TOP-K SELECTION (model_selection_method = "top_k")
#   - Select the k best models ranked by total error
#   - Set num_top_models to desired ensemble size
#
# Method 2: THRESHOLD SELECTION (model_selection_method = "threshold")
#   - Select all models meeting error criteria
#   - Models must satisfy: relative error < threshold for mean AND std
#   - Thresholds defined separately for Gamma_n and Gamma_c
# =============================================================================
model_selection_method = "threshold"  # Options: "top_k" or "threshold"

# Top-K parameters
num_top_models = 20               # Number of best models to keep

# Threshold parameters (used when model_selection_method = "threshold")
# Relative error thresholds: |pred - ref| / |ref| < threshold
threshold_mean_error = 0.05       # Maximum relative error in mean (5%)
threshold_std_error = 0.30        # Maximum relative error in std (30%)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def bprint(string: str):
    """Print bold text to console."""
    print("\033[1m" + string + "\033[0m")

def get_memmap_path(name):
    """Get path for memory-mapped file."""
    return os.path.join(output_path, f"memmap_{name}.dat")

def cleanup_memmap(name):
    """Remove memory-mapped file if it exists."""
    path = get_memmap_path(name)
    if os.path.exists(path):
        os.remove(path)

def get_dt_from_file(file_path, default=0.025):
    """Extract dt from h5 file attributes. Returns default if not found."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Check common attribute locations
            if 'dt' in f.attrs:
                return float(f.attrs['dt'])
            if 'dt' in f:
                return float(f['dt'][()])
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

def compute_truncation_snapshots(file_path, truncate_snapshots, truncate_time, default_dt):
    """Compute number of snapshots to keep based on truncation settings."""
    if truncate_snapshots is not None:
        return truncate_snapshots
    elif truncate_time is not None:
        dt = get_dt_from_file(file_path, default_dt)
        n_snaps = int(truncate_time / dt)
        print(f"    Using dt={dt:.4f} -> {n_snaps} snapshots for t={truncate_time}")
        return n_snaps
    return None

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
if cluster:
    bprint("Using cluster settings")
    from config.cluster import *
else:
    bprint("Using local settings")
    from config.local import *

if r > svd_save:
    bprint("Warning! r value larger than svd_save")

# =============================================================================
# PRINT CONFIGURATION SUMMARY
# =============================================================================
bprint(f"Number of training trajectories: {len(training_files)}")
bprint(f"Number of test trajectories: {len(test_files)}")

if truncate_data:
    if truncate_snapshots is not None:
        bprint(f"Data truncation: {truncate_snapshots} snapshots per trajectory")
    elif truncate_time is not None:
        bprint(f"Data truncation: {truncate_time} time units (dt={default_dt})")
else:
    bprint("Data truncation: DISABLED")

bprint(f"Model selection method: {model_selection_method}")
if model_selection_method == "top_k":
    print(f"  -> Selecting top {num_top_models} models by total error")
elif model_selection_method == "threshold":
    print(f"  -> Selecting models with mean error < {threshold_mean_error:.1%}")
    print(f"  -> Selecting models with std error < {threshold_std_error:.1%}")

# %%
np.__config__.show()

# %% [markdown]
# ## Step 1: Compute POD basis

# %% [markdown]
# ### Step 1.1: Load training and test data

# %%
def load_and_process_snapshots(file_path, i, dataset_name="", max_snapshots=None):
    """Helper function to load and process a single snapshot file.
    
    Args:
        file_path: Path to the h5 file
        i: Index of this file (for printing)
        dataset_name: Prefix for print statements
        max_snapshots: If set, truncate to this many snapshots
        
    Returns processed array and its shape info (doesn't append to list).
    """
    print(f"  Loading IC {i+1}: {file_path}")
    fh = xr.open_dataset(file_path, 
                         engine=ENGINE, 
                         phony_dims="sort"
                        )
    
    # Get density and phi as numpy arrays
    density = fh["density"].values
    phi = fh["phi"].values
    fh.close()  # Explicitly close file handle
    
    # Apply truncation if specified
    original_n_time = density.shape[0]
    if max_snapshots is not None and max_snapshots < original_n_time:
        density = density[:max_snapshots]
        phi = phi[:max_snapshots]
        print(f"    Truncated: {original_n_time} -> {max_snapshots} snapshots")
    
    if i == 0 and show_plots:
        plt.imshow(density[0])
        plt.colorbar()
        plt.show()
        bprint(f"{dataset_name}IC{i} shape: {density.shape}")
    
    # If already 3D (time, y, x), use as-is. If 2D (time, flattened), reshape to 3D
    if density.ndim == 2:
        n_time = density.shape[0]
        grid_size = int(np.sqrt(density.shape[1]))
        density = density.reshape(n_time, grid_size, grid_size)
        phi = phi.reshape(n_time, grid_size, grid_size)
    
    # Stack fields: (time, y, x) -> (field, time, y, x)
    Q_ic = np.stack([density, phi], axis=0)  # Shape: (2, time, y, x)
    del density, phi  # Free memory immediately
    
    # Transpose to (field, y, x, time) before flattening
    Q_ic = Q_ic.transpose(0, 2, 3, 1)  # Shape: (2, 256, 256, time)
    
    # Now reshape: flatten spatial dimensions
    n_field, n_y, n_x, n_time = Q_ic.shape
    Q_ic = Q_ic.reshape(n_field * n_y * n_x, n_time)  # Shape: (131072, time)
    
    print(f"    Shape: {Q_ic.shape}")
    return Q_ic


if step_1:
    bprint("Reading snapshot(s) - Building memory-mapped arrays")
    
    # First pass: determine total sizes (with truncation)
    train_timesteps = []
    test_timesteps = []
    train_truncations = []  # Store truncation for each training file
    test_truncations = []   # Store truncation for each test file
    n_spatial = None
    
    for file_path in training_files:
        with xr.open_dataset(file_path, engine=ENGINE, phony_dims="sort") as fh:
            n_time_original = fh["density"].shape[0]
            if n_spatial is None:
                if fh["density"].ndim == 3:
                    n_spatial = 2 * fh["density"].shape[1] * fh["density"].shape[2]
                else:
                    n_spatial = 2 * fh["density"].shape[1]
        
        # Compute truncation for this file
        if truncate_data:
            max_snaps = compute_truncation_snapshots(
                file_path, truncate_snapshots, truncate_time, default_dt
            )
            if max_snaps is not None:
                n_time = min(n_time_original, max_snaps)
            else:
                n_time = n_time_original
        else:
            n_time = n_time_original
            max_snaps = None
        
        train_timesteps.append(n_time)
        train_truncations.append(max_snaps)
    
    for file_path in test_files:
        with xr.open_dataset(file_path, engine=ENGINE, phony_dims="sort") as fh:
            n_time_original = fh["density"].shape[0]
        
        # Compute truncation for this file
        if truncate_data:
            max_snaps = compute_truncation_snapshots(
                file_path, truncate_snapshots, truncate_time, default_dt
            )
            if max_snaps is not None:
                n_time = min(n_time_original, max_snaps)
            else:
                n_time = n_time_original
        else:
            n_time = n_time_original
            max_snaps = None
        
        test_timesteps.append(n_time)
        test_truncations.append(max_snaps)
    
    total_train_time = sum(train_timesteps)
    total_test_time = sum(test_timesteps)
    
    bprint(f"Creating memory-mapped arrays: {n_spatial} spatial × {total_train_time} train timesteps")
    if truncate_data:
        print(f"  (Truncation applied: {train_timesteps} train, {test_timesteps} test)")
    
    # Create memory-mapped arrays for Q_train and Q_test
    cleanup_memmap("Q_train")
    cleanup_memmap("Q_test")
    
    Q_train = np.memmap(get_memmap_path("Q_train"), dtype='float64', mode='w+', 
                        shape=(n_spatial, total_train_time))
    Q_test = np.memmap(get_memmap_path("Q_test"), dtype='float64', mode='w+', 
                       shape=(n_spatial, total_test_time))
    
    # Store timestep boundaries for later IC extraction
    train_boundaries = [0] + list(np.cumsum(train_timesteps))
    test_boundaries = [0] + list(np.cumsum(test_timesteps))
    
    # Load training data directly into memmap
    for i, file_path in enumerate(training_files):
        Q_ic = load_and_process_snapshots(
            file_path, i, dataset_name="", 
            max_snapshots=train_truncations[i]
        )
        start_idx = train_boundaries[i]
        end_idx = train_boundaries[i + 1]
        Q_train[:, start_idx:end_idx] = Q_ic
        
        if i == 0 and show_plots:
            img = Q_ic[:256*256, 0].reshape(256, 256)
            plt.imshow(img)
            plt.colorbar()
            plt.show()
        del Q_ic
        gc.collect()
    
    bprint(f"Combined training data shape: {Q_train.shape}")
    
    # Load test data directly into memmap
    for i, file_path in enumerate(test_files):
        Q_ic = load_and_process_snapshots(
            file_path, i, dataset_name="Test ",
            max_snapshots=test_truncations[i]
        )
        start_idx = test_boundaries[i]
        end_idx = test_boundaries[i + 1]
        Q_test[:, start_idx:end_idx] = Q_ic
        del Q_ic
        gc.collect()
    
    bprint(f"Combined test data shape: {Q_test.shape}")
    
    # Save boundaries for IC extraction (include truncation info)
    np.savez(output_path + "data_boundaries.npz", 
             train_boundaries=train_boundaries, 
             test_boundaries=test_boundaries,
             n_spatial=n_spatial,
             train_timesteps=train_timesteps,
             test_timesteps=test_timesteps,
             truncate_data=truncate_data,
             truncate_snapshots=truncate_snapshots if truncate_snapshots else -1,
             truncate_time=truncate_time if truncate_time else -1.0)
    
else:
    bprint("Skipping data loading (step_1=False, will load POD directly)")

# %% [markdown]
# ### Step 1.2: Compute POD

# %%
if step_1: 
    # Compute POD basis from combined training data
    bprint("Computing POD basis from all training trajectories...")
    start_time = time.time()

    U, S, _ = np.linalg.svd(Q_train, full_matrices=False)

    elapsed = time.time() - start_time
    print(f"  POD computation completed in {elapsed:.3f} seconds.")
    
    # Save POD data
    POD_file_multi = output_path + "POD_multi_IC.npz"
    np.savez(POD_file_multi, S=S, U=U)
    print(f"  Saved POD basis to {POD_file_multi}")
    print(f"  U shape: {U.shape}, S shape: {S.shape}")
    
    gc.collect()

else:
    # Load previous POD basis
    bprint("Loading POD basis...")
    POD_file_multi = output_path + "POD_multi_IC.npz"
    POD_multi = np.load(POD_file_multi)
    S, U = POD_multi['S'], POD_multi['U']
    del POD_multi
    gc.collect()
    print(f"  Loaded POD basis from {POD_file_multi}")
    print(f"  U shape: {U.shape}, S shape: {S.shape}")

# %% [markdown]
# ### Step 1.3: Project train & test

# %%
if step_1:
    # Project training data
    bprint("Projecting training data...")
    # Use only the modes we need (truncate U to r modes for projection storage)
    Ur = U[:, :r]
    
    Xhat_train = Q_train.T @ Ur  # Shape: (n_time, r) - much smaller!
    Xhat_train_file = output_path + "X_hat_train_multi_IC.npy"
    np.save(Xhat_train_file, Xhat_train)
    print(f"  Saved to {Xhat_train_file}, shape: {Xhat_train.shape}")
    
    # Project test data
    bprint("Projecting test data...")
    Xhat_test = Q_test.T @ Ur
    Xhat_test_file = output_path + "X_hat_test_multi_IC.npy"
    np.save(Xhat_test_file, Xhat_test)
    print(f"  Saved to {Xhat_test_file}, shape: {Xhat_test.shape}")
    
    del Ur
    gc.collect()

else:
    # Load pre-computed projections
    bprint("Loading pre-computed projections...")
    Xhat_train_file = output_path + "X_hat_train_multi_IC.npy"
    Xhat_test_file = output_path + "X_hat_test_multi_IC.npy"
    
    Xhat_train = np.load(Xhat_train_file)
    Xhat_test = np.load(Xhat_test_file)
    print(f"  Loaded Xhat_train: {Xhat_train.shape}")
    print(f"  Loaded Xhat_test: {Xhat_test.shape}")

# %%
if step_1:
    # Save initial conditions for later use
    bprint("Saving initial conditions...")
    
    # Load boundaries
    boundaries = np.load(output_path + "data_boundaries.npz")
    train_boundaries = boundaries['train_boundaries']
    test_boundaries = boundaries['test_boundaries']
    
    # Extract ICs from memmap (only first timestep of each trajectory)
    train_ICs = np.array([Q_train[:, train_boundaries[i]] for i in range(len(training_files))])
    test_ICs = np.array([Q_test[:, test_boundaries[i]] for i in range(len(test_files))])
    
    # Reduced ICs from projected data
    train_ICs_reduced = np.array([Xhat_train[train_boundaries[i], :] for i in range(len(training_files))])
    test_ICs_reduced = np.array([Xhat_test[test_boundaries[i], :] for i in range(len(test_files))])
    
    np.savez(
        output_path + "initial_conditions_multi_IC.npz",
        train_ICs=train_ICs,
        test_ICs=test_ICs,
        train_ICs_reduced=train_ICs_reduced,
        test_ICs_reduced=test_ICs_reduced
    )
    print(f"  Saved ICs: train_ICs {train_ICs.shape}, test_ICs {test_ICs.shape}")
    
    del train_ICs, test_ICs, train_ICs_reduced, test_ICs_reduced
    
    # Clean up large memory-mapped arrays - we're done with them
    bprint("Cleaning up large arrays...")
    del Q_train, Q_test
    gc.collect()
    
    # Optionally remove memmap files to free disk space
    # cleanup_memmap("Q_train")
    # cleanup_memmap("Q_test")
    
    bprint("Done with Step 1.")

else:
    bprint("Skipping IC saving (step_1=False)")

# %% [markdown]
# ## Step 2: Compute ROM

# %%
def solve_opinf_difference_model(s0, n_steps, f):
    s = np.zeros((np.size(s0), n_steps))
    is_nan = False

    s[:, 0] = s0
    for i in range(n_steps - 1):
        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):
            print("NaN encountered at iteration " + str(i + 1))
            is_nan = True
            break

    return is_nan, s

# %% [markdown]
# ### Step 2.1: Prep data
# 
# **Important: Handling Multiple Initial Conditions Correctly**
# 
# When learning the **state evolution model** ($\hat{x}_{k+1} = f(\hat{x}_k)$), we must avoid creating false transitions between trajectories:
# 
# | Approach | What Happens |
# |----------|--------------|
# | ❌ **Wrong**: `X_state = Xhat_all[:-1]`, `Y_state = Xhat_all[1:]` | Creates a fake transition from last state of traj A → first state of traj B |
# | ✅ **Correct**: Create pairs within each trajectory, then stack | Each `(X_state[i], Y_state[i])` pair is a valid physical transition |
# 
# For the **output model** ($y = g(\hat{x})$), simple concatenation is fine because each sample is independent (no temporal dependency).

# %%
bprint("Prepare the data for learning...")

# Ensure we have Xhat_train loaded
if 'Xhat_train' not in dir() or Xhat_train is None:
    bprint("Loading Xhat_train from disk...")
    Xhat_train = np.load(output_path + "X_hat_train_multi_IC.npy")

print(f"Training data shape: {Xhat_train.shape}")

# Truncate to r modes if needed (in case we loaded full projection)
if Xhat_train.shape[1] > r:
    Xhat_train = Xhat_train[:, :r]
    print(f"  Truncated to r={r} modes: {Xhat_train.shape}")

# Load trajectory boundaries to handle multiple ICs correctly
boundaries_data = np.load(output_path + "data_boundaries.npz")
train_boundaries = boundaries_data['train_boundaries']
n_train_traj = len(train_boundaries) - 1

print(f"  Number of training trajectories: {n_train_traj}")
print(f"  Trajectory boundaries: {train_boundaries}")

# CORRECT APPROACH: Create input-output pairs WITHIN each trajectory, then concatenate
# This avoids creating false transitions between the end of one trajectory 
# and the start of the next
X_state_list = []
Y_state_list = []

for traj_idx in range(n_train_traj):
    start_idx = train_boundaries[traj_idx]
    end_idx = train_boundaries[traj_idx + 1]
    
    # Extract this trajectory's data
    Xhat_traj = Xhat_train[start_idx:end_idx, :]
    
    # Create valid input-output pairs within this trajectory
    X_state_traj = Xhat_traj[:-1, :]  # States k (exclude last)
    Y_state_traj = Xhat_traj[1:, :]   # States k+1 (exclude first)
    
    X_state_list.append(X_state_traj)
    Y_state_list.append(Y_state_traj)
    
    print(f"    Trajectory {traj_idx + 1}: {Xhat_traj.shape[0]} timesteps -> {X_state_traj.shape[0]} valid pairs")

# Stack all valid pairs (no false transitions between trajectories!)
X_state = np.vstack(X_state_list)
Y_state = np.vstack(Y_state_list)

print(f"\n  Total valid state pairs: {X_state.shape[0]}")
print(f"  (Note: This is {n_train_traj} fewer than naive approach due to excluding trajectory boundaries)")

s = int(r * (r + 1) / 2)
d_state = r + s
d_out = r + s + 1

X_state2 = get_x_sq(X_state)
D_state = np.concatenate((X_state, X_state2), axis=1)
D_state_2 = D_state.T @ D_state
bprint("State learning data prepared")

# %%
bprint("Prepare the output learning data")

# NOTE: For output learning, we use ALL timesteps (not pairs), so simple concatenation is correct
# Each row is an independent sample: X_out[k] -> Y_Gamma[k] (no temporal dependency)
X_out = Xhat_train
K = X_out.shape[0]
E = np.ones((K, 1))

mean_Xhat = np.mean(X_out, axis=0)
Xhat_out = X_out - mean_Xhat[np.newaxis, :]

local_min = np.min(X_out)
local_max = np.max(X_out)
local_scaling = np.maximum(np.abs(local_min), np.abs(local_max))
scaling_Xhat = local_scaling

Xhat_out /= scaling_Xhat
Xhat_out2 = get_x_sq(Xhat_out)

D_out = np.concatenate((Xhat_out, Xhat_out2, E), axis=1)
D_out_2 = D_out.T @ D_out

print(f"D_out shape: {D_out.shape}")
print(f"D_out_2 condition number: {np.linalg.cond(D_out_2):.2e}")
bprint("Done")

# %%
bprint("Load derived quantities from all training trajectories")

Gamma_n_list = []
Gamma_c_list = []

for file_path in training_files:
    fh = loader(file_path, ENGINE=ENGINE)
    Gamma_n_list.append(fh["gamma_n"].data)
    Gamma_c_list.append(fh["gamma_c"].data)

# Concatenate all trajectories
Gamma_n = np.concatenate(Gamma_n_list)
Gamma_c = np.concatenate(Gamma_c_list)

mean_Gamma_n_ref = np.mean(Gamma_n)
std_Gamma_n_ref = np.std(Gamma_n, ddof=1)

mean_Gamma_c_ref = np.mean(Gamma_c)
std_Gamma_c_ref = np.std(Gamma_c, ddof=1)

Y_Gamma = np.vstack((Gamma_n, Gamma_c))

print(f"Gamma_n shape: {Gamma_n.shape}")
print(f"Gamma_c shape: {Gamma_c.shape}")
print(f"Y_Gamma shape: {Y_Gamma.shape}")
print(f"X_out shape (for output learning): {X_out.shape}")
print(f"Shape compatibility check: Y_Gamma cols ({Y_Gamma.shape[1]}) vs X_out rows ({X_out.shape[0]})")

if Y_Gamma.shape[1] != X_out.shape[0]:
    raise ValueError(f"Shape mismatch: Y_Gamma has {Y_Gamma.shape[1]} columns but X_out has {X_out.shape[0]} rows")

print(f"Mean Gamma_n: {mean_Gamma_n_ref:.4f}, Std: {std_Gamma_n_ref:.4f}")
print(f"Mean Gamma_c: {mean_Gamma_c_ref:.4f}, Std: {std_Gamma_c_ref:.4f}")
print("Done")

# %% [markdown]
# ### Step 2.2: Model Regularization Sweep

# %%
import heapq

class TopKModels:
    """
    Maintains a collection of the k best models based on score (lower is better).
    Uses a min-heap for efficient top-k tracking.
    """
    def __init__(self, k=5):
        self.k = k
        self.heap = []  # (score, model_info)

    def add(self, score, model):
        entry = (score, model)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
        else:
            if score < self.heap[0][0]:
                heapq.heapreplace(self.heap, entry)

    def get_best(self):
        """Return models sorted from best to worst."""
        return sorted(self.heap, reverse=False)


class ThresholdModels:
    """
    Collects all models meeting specified error thresholds.
    
    Selection criteria:
        - mean_err_Gamma_n < threshold_mean
        - std_err_Gamma_n < threshold_std
        - mean_err_Gamma_c < threshold_mean
        - std_err_Gamma_c < threshold_std
    """
    def __init__(self, threshold_mean=0.05, threshold_std=0.30):
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.models = []

    def add(self, model):
        """Add model if it meets all threshold criteria."""
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

    def get_best(self):
        """Return models sorted from best to worst by total error."""
        return sorted(self.models, key=lambda x: x[0])


# Initialize model collector based on selection method
if model_selection_method == "top_k":
    model_collector = TopKModels(k=num_top_models)
    bprint(f"Model selection: Top-{num_top_models} by total error")
elif model_selection_method == "threshold":
    model_collector = ThresholdModels(
        threshold_mean=threshold_mean_error,
        threshold_std=threshold_std_error
    )
    bprint(f"Model selection: Threshold-based")
    print(f"  Mean error threshold: {threshold_mean_error:.1%}")
    print(f"  Std error threshold: {threshold_std_error:.1%}")
else:
    raise ValueError(f"Unknown model_selection_method: {model_selection_method}")

# %%
from tqdm import tqdm

if step_2:
    bprint("BEGIN HYPERPARAMETER SWEEP")
    print(f"  State regularization: {len(ridge_alf_lin_all)} x {len(ridge_alf_quad_all)} combinations")
    print(f"  Output regularization: {len(gamma_reg_lin)} x {len(gamma_reg_quad)} combinations")
    
    n_total_combinations = (len(ridge_alf_lin_all) * len(ridge_alf_quad_all) * 
                           len(gamma_reg_lin) * len(gamma_reg_quad))
    print(f"  Total combinations to evaluate: {n_total_combinations}\n")
    
    sweep_start_time = time.time()
    n_evaluated = 0
    n_nan_models = 0
    n_accepted = 0
    best_error_so_far = float('inf')
    
    # Create progress bar
    pbar = tqdm(total=n_total_combinations, 
                desc="Evaluating models",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for alpha_state_lin in ridge_alf_lin_all:
        for alpha_state_quad in ridge_alf_quad_all:
            # Construct and solve regularized state operator learning problem
            regg = np.zeros(d_state)
            regg[:r] = alpha_state_lin
            regg[r:r + s] = alpha_state_quad
            regularizer = np.diag(regg)
            D_state_reg = D_state_2 + regularizer

            O = np.linalg.solve(D_state_reg, np.dot(D_state.T, Y_state)).T

            A = O[:, :r]
            F = O[:, r:r + s]
            
            # Define state evolution function (closure captures current A, F)
            A_current, F_current = A.copy(), F.copy()
            f = lambda x, A=A_current, F=F_current: np.dot(A, x) + np.dot(F, get_x_sq(x))

            # Integrate learned model forward in time
            u0 = X_state[0, :]
            is_nan, Xhat_pred = solve_opinf_difference_model(u0, n_steps, f)
            
            if is_nan:
                n_nan_models += 1
                # Still update progress for skipped combinations
                pbar.update(len(gamma_reg_lin) * len(gamma_reg_quad))
                continue
                
            X_OpInf_full = Xhat_pred.T
            
            # Prepare predicted states for output operator learning
            Xhat_OpInf_scaled = (X_OpInf_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
            Xhat_2_OpInf = get_x_sq(Xhat_OpInf_scaled)
            
            # Sweep over output operator regularization
            for alpha_out_lin in gamma_reg_lin:
                for alpha_out_quad in gamma_reg_quad:
                    n_evaluated += 1
                    
                    # Construct and solve regularized output operator learning problem
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

                    # Compute output predictions
                    Y_OpInf = (
                        C @ Xhat_OpInf_scaled.T
                        + G @ Xhat_2_OpInf.T
                        + c[:, np.newaxis]
                    )

                    ts_Gamma_n = Y_OpInf[0, :]
                    ts_Gamma_c = Y_OpInf[1, :]
                    
                    # Compute statistical error metrics on training portion
                    mean_Gamma_n_OpInf = np.mean(ts_Gamma_n[:training_end])
                    std_Gamma_n_OpInf = np.std(ts_Gamma_n[:training_end], ddof=1)
                    mean_Gamma_c_OpInf = np.mean(ts_Gamma_c[:training_end])
                    std_Gamma_c_OpInf = np.std(ts_Gamma_c[:training_end], ddof=1)
                    
                    # Relative errors
                    mean_err_Gamma_n = np.abs(mean_Gamma_n_ref - mean_Gamma_n_OpInf) / np.abs(mean_Gamma_n_ref)
                    std_err_Gamma_n = np.abs(std_Gamma_n_ref - std_Gamma_n_OpInf) / std_Gamma_n_ref
                    mean_err_Gamma_c = np.abs(mean_Gamma_c_ref - mean_Gamma_c_OpInf) / np.abs(mean_Gamma_c_ref)
                    std_err_Gamma_c = np.abs(std_Gamma_c_ref - std_Gamma_c_OpInf) / std_Gamma_c_ref
                    
                    # Aggregate error metric
                    total_error = (mean_err_Gamma_n + std_err_Gamma_n + 
                                   mean_err_Gamma_c + std_err_Gamma_c)

                    # Update best error tracker
                    if total_error < best_error_so_far:
                        best_error_so_far = total_error
                        pbar.set_postfix({'best_err': f'{best_error_so_far:.4e}', 
                                         'NaNs': n_nan_models}, refresh=False)

                    # Store model with all metadata
                    model = {
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
                        'alpha_out_quad': alpha_out_quad
                    }

                    # Add to collector based on selection method
                    if model_selection_method == "top_k":
                        model_collector.add(score=total_error, model=model)
                    elif model_selection_method == "threshold":
                        if model_collector.add(model):
                            n_accepted += 1
                    
                    # Update progress bar
                    pbar.update(1)
    
    pbar.close()
    sweep_elapsed = time.time() - sweep_start_time
    
    print()  # Blank line after progress bar
    bprint("HYPERPARAMETER SWEEP COMPLETE")
    print(f"  Total time: {sweep_elapsed:.1f} seconds ({sweep_elapsed/60:.1f} minutes)")
    print(f"  Models evaluated: {n_evaluated}")
    print(f"  Models with NaN: {n_nan_models}")
    
    # Retrieve selected models
    best_models = model_collector.get_best()
    
    if model_selection_method == "top_k":
        print(f"  Models selected (top-{num_top_models}): {len(best_models)}")
    elif model_selection_method == "threshold":
        print(f"  Models meeting threshold criteria: {len(best_models)}")
    
    if len(best_models) > 0:
        print(f"\n  Best model total error: {best_models[0][0]:.6e}")
        print(f"  Worst selected model total error: {best_models[-1][0]:.6e}")
        
        # Summary table of selected models
        print(f"\n  {'='*70}")
        print(f"  {'Model':>6} | {'Total Err':>10} | {'Mean Γn':>8} | {'Std Γn':>8} | {'Mean Γc':>8} | {'Std Γc':>8}")
        print(f"  {'-'*70}")
        for i, (score, model) in enumerate(best_models[:10]):  # Show top 10
            print(f"  {i+1:>6} | {score:>10.4e} | {model['mean_err_Gamma_n']:>8.4f} | "
                  f"{model['std_err_Gamma_n']:>8.4f} | {model['mean_err_Gamma_c']:>8.4f} | "
                  f"{model['std_err_Gamma_c']:>8.4f}")
        if len(best_models) > 10:
            print(f"  {'...'}")
            print(f"  (showing 10 of {len(best_models)} models)")
        print(f"  {'='*70}")
    else:
        bprint("WARNING: No models met selection criteria!")
        if model_selection_method == "threshold":
            print("  Consider relaxing threshold_mean_error or threshold_std_error")
    
else:
    bprint("Skipping model search (step_2=False)")

# %%
if step_2:
    bprint("Saving ensemble models...")
    
    if best_models is not None and len(best_models) > 0:
        # Construct filename with selection method info
        if model_selection_method == "top_k":
            model_filename = f"ensemble_models_r{r}_topk{len(best_models)}.npz"
        else:
            model_filename = f"ensemble_models_r{r}_thresh{len(best_models)}.npz"
        
        # Build ensemble data dictionary
        ensemble_data = {
            'num_models': len(best_models),
            'selection_method': model_selection_method,
            'r': r
        }
        
        # Store threshold parameters if using threshold method
        if model_selection_method == "threshold":
            ensemble_data['threshold_mean_error'] = threshold_mean_error
            ensemble_data['threshold_std_error'] = threshold_std_error
        else:
            ensemble_data['num_top_models'] = num_top_models
        
        # Save each model's operators and hyperparameters
        for i, (score, model) in enumerate(best_models):
            prefix = f'model_{i}_'
            ensemble_data[prefix + 'A'] = model['A']
            ensemble_data[prefix + 'F'] = model['F']
            ensemble_data[prefix + 'C'] = model['C']
            ensemble_data[prefix + 'G'] = model['G']
            ensemble_data[prefix + 'c'] = model['c']
            ensemble_data[prefix + 'alpha_state_lin'] = model['alpha_state_lin']
            ensemble_data[prefix + 'alpha_state_quad'] = model['alpha_state_quad']
            ensemble_data[prefix + 'alpha_out_lin'] = model['alpha_out_lin']
            ensemble_data[prefix + 'alpha_out_quad'] = model['alpha_out_quad']
            ensemble_data[prefix + 'total_error'] = model['total_error']
            ensemble_data[prefix + 'mean_err_Gamma_n'] = model['mean_err_Gamma_n']
            ensemble_data[prefix + 'std_err_Gamma_n'] = model['std_err_Gamma_n']
            ensemble_data[prefix + 'mean_err_Gamma_c'] = model['mean_err_Gamma_c']
            ensemble_data[prefix + 'std_err_Gamma_c'] = model['std_err_Gamma_c']
        
        np.savez(output_path + model_filename, **ensemble_data)
        bprint(f"Ensemble saved: {output_path}{model_filename}")
        print(f"  Selection method: {model_selection_method}")
        print(f"  Number of models: {len(best_models)}")
        
    else:
        bprint("WARNING: No valid models found during sweep!")
        
else:
    bprint("Loading pre-computed ensemble models...")
    
    # Try to find existing ensemble file
    # First try the current selection method, then try alternatives
    if model_selection_method == "top_k":
        primary_file = output_path + f"ensemble_models_r{r}_topk{num_top_models}.npz"
        alt_pattern = f"ensemble_models_r{r}_topk*.npz"
    else:
        primary_file = output_path + f"ensemble_models_r{r}_thresh*.npz"
        alt_pattern = f"ensemble_models_r{r}_thresh*.npz"
    
    # Also check legacy filename format
    legacy_file = output_path + f"ensemble_models_r{r}_k{num_top_models}.npz"
    
    model_file = None
    if os.path.exists(primary_file):
        model_file = primary_file
    elif os.path.exists(legacy_file):
        model_file = legacy_file
        print(f"  Using legacy file format: {legacy_file}")
    else:
        # Search for any matching ensemble file
        import glob
        candidates = glob.glob(output_path + f"ensemble_models_r{r}_*.npz")
        if candidates:
            model_file = candidates[0]
            print(f"  Found ensemble file: {model_file}")
    
    if model_file is None or not os.path.exists(model_file):
        bprint(f"ERROR: No ensemble model file found for r={r}")
        best_models = None
    else:
        ensemble_data = np.load(model_file, allow_pickle=True)
        num_loaded = int(ensemble_data['num_models'])
        
        # Read selection method metadata if available
        loaded_selection_method = str(ensemble_data.get('selection_method', 'top_k'))
        
        best_models = []
        for i in range(num_loaded):
            prefix = f'model_{i}_'
            model = {
                'A': ensemble_data[prefix + 'A'],
                'F': ensemble_data[prefix + 'F'],
                'C': ensemble_data[prefix + 'C'],
                'G': ensemble_data[prefix + 'G'],
                'c': ensemble_data[prefix + 'c'],
                'total_error': float(ensemble_data[prefix + 'total_error']),
                'mean_err_Gamma_n': float(ensemble_data[prefix + 'mean_err_Gamma_n']),
                'std_err_Gamma_n': float(ensemble_data[prefix + 'std_err_Gamma_n']),
                'mean_err_Gamma_c': float(ensemble_data[prefix + 'mean_err_Gamma_c']),
                'std_err_Gamma_c': float(ensemble_data[prefix + 'std_err_Gamma_c']),
                'alpha_state_lin': float(ensemble_data[prefix + 'alpha_state_lin']),
                'alpha_state_quad': float(ensemble_data[prefix + 'alpha_state_quad']),
                'alpha_out_lin': float(ensemble_data[prefix + 'alpha_out_lin']),
                'alpha_out_quad': float(ensemble_data[prefix + 'alpha_out_quad'])
            }
            score = model['total_error']
            best_models.append((score, model))
        
        print(f"  Loaded {num_loaded} models from: {model_file}")
        print(f"  Selection method used: {loaded_selection_method}")
        print(f"  Best model total error: {best_models[0][0]:.6e}")

# %% [markdown]
# ## Step 3: Make Predictions with model

# %%
if best_models is not None and len(best_models) > 0:
    bprint(f"Computing ensemble predictions with {len(best_models)} models...")
    
    # Load initial conditions and boundaries
    IC_data = np.load(output_path + "initial_conditions_multi_IC.npz")
    boundaries_data = np.load(output_path + "data_boundaries.npz")
    
    train_ICs_reduced = IC_data['train_ICs_reduced']  # Shape: (n_train_traj, r)
    test_ICs_reduced = IC_data['test_ICs_reduced']    # Shape: (n_test_traj, r)
    train_boundaries = boundaries_data['train_boundaries']
    test_boundaries = boundaries_data['test_boundaries']
    
    n_train_traj = len(train_boundaries) - 1
    n_test_traj = len(test_boundaries) - 1
    
    print(f"  Number of training trajectories: {n_train_traj}")
    print(f"  Number of test trajectories: {n_test_traj}")
    
    # Storage for ensemble predictions
    # For each trajectory, we'll store predictions from all models
    train_predictions = {
        'Gamma_n': [],  # List of [n_models, n_timesteps] arrays, one per trajectory
        'Gamma_c': [],
        'X_OpInf': []
    }
    
    test_predictions = {
        'Gamma_n': [],
        'Gamma_c': [],
        'X_OpInf': []
    }
    
    ###########################
    # TRAINING TRAJECTORIES
    ###########################
    bprint("\nProcessing TRAINING trajectories...")
    for traj_idx in range(n_train_traj):
        print(f"\n  Training trajectory {traj_idx + 1}/{n_train_traj}")
        
        # Get this trajectory's length
        traj_length = train_boundaries[traj_idx + 1] - train_boundaries[traj_idx]
        
        # Storage for all models' predictions for this trajectory
        traj_Gamma_n_preds = []
        traj_Gamma_c_preds = []
        traj_X_OpInf_preds = []
        
        # Initial condition for this trajectory
        u0 = train_ICs_reduced[traj_idx, :]
        
        # Run each model
        for model_idx, (score, model) in enumerate(best_models):
            # Extract operators
            A_model = model['A']
            F_model = model['F']
            C_model = model['C']
            G_model = model['G']
            c_model = model['c']
            
            # State evolution function
            f = lambda x: np.dot(A_model, x) + np.dot(F_model, get_x_sq(x))
            
            # Solve for this trajectory length
            is_nan, Xhat_pred = solve_opinf_difference_model(u0, traj_length, f)
            
            if is_nan:
                print(f"    WARNING: NaN in model {model_idx + 1}, skipping...")
                continue
            
            X_OpInf_full = Xhat_pred.T  # Shape: (traj_length, r)
            
            # Apply output operators
            Xhat_OpInf_scaled = (X_OpInf_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
            Xhat_2_OpInf = get_x_sq(Xhat_OpInf_scaled)
            
            Y_OpInf = (
                C_model @ Xhat_OpInf_scaled.T
                + G_model @ Xhat_2_OpInf.T
                + c_model[:, np.newaxis]
            )
            
            traj_Gamma_n_preds.append(Y_OpInf[0, :])
            traj_Gamma_c_preds.append(Y_OpInf[1, :])
            traj_X_OpInf_preds.append(X_OpInf_full)
        
        # Convert to arrays
        train_predictions['Gamma_n'].append(np.array(traj_Gamma_n_preds))  # Shape: (n_models, traj_length)
        train_predictions['Gamma_c'].append(np.array(traj_Gamma_c_preds))
        train_predictions['X_OpInf'].append(np.array(traj_X_OpInf_preds))  # Shape: (n_models, traj_length, r)
    
    ###########################
    # TEST TRAJECTORIES
    ###########################
    bprint("\nProcessing TEST trajectories...")
    for traj_idx in range(n_test_traj):
        print(f"\n  Test trajectory {traj_idx + 1}/{n_test_traj}")
        
        # Get this trajectory's length
        traj_length = test_boundaries[traj_idx + 1] - test_boundaries[traj_idx]
        
        # Storage for all models' predictions for this trajectory
        traj_Gamma_n_preds = []
        traj_Gamma_c_preds = []
        traj_X_OpInf_preds = []
        
        # Initial condition for this trajectory
        u0 = test_ICs_reduced[traj_idx, :]
        
        # Run each model
        for model_idx, (score, model) in enumerate(best_models):
            # Extract operators
            A_model = model['A']
            F_model = model['F']
            C_model = model['C']
            G_model = model['G']
            c_model = model['c']
            
            # State evolution function
            f = lambda x: np.dot(A_model, x) + np.dot(F_model, get_x_sq(x))
            
            # Solve for this trajectory length
            is_nan, Xhat_pred = solve_opinf_difference_model(u0, traj_length, f)
            
            if is_nan:
                print(f"    WARNING: NaN in model {model_idx + 1}, skipping...")
                continue
            
            X_OpInf_full = Xhat_pred.T  # Shape: (traj_length, r)
            
            # Apply output operators
            Xhat_OpInf_scaled = (X_OpInf_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
            Xhat_2_OpInf = get_x_sq(Xhat_OpInf_scaled)
            
            Y_OpInf = (
                C_model @ Xhat_OpInf_scaled.T
                + G_model @ Xhat_2_OpInf.T
                + c_model[:, np.newaxis]
            )
            
            traj_Gamma_n_preds.append(Y_OpInf[0, :])
            traj_Gamma_c_preds.append(Y_OpInf[1, :])
            traj_X_OpInf_preds.append(X_OpInf_full)
        
        # Convert to arrays
        test_predictions['Gamma_n'].append(np.array(traj_Gamma_n_preds))
        test_predictions['Gamma_c'].append(np.array(traj_Gamma_c_preds))
        test_predictions['X_OpInf'].append(np.array(traj_X_OpInf_preds))
    
    ###########################
    # Save predictions
    ###########################
    bprint("\nSaving ensemble trajectory predictions...")
    
    save_dict = {
        'n_train_traj': n_train_traj,
        'n_test_traj': n_test_traj,
        'num_models_used': len(best_models),
        'train_boundaries': train_boundaries,
        'test_boundaries': test_boundaries
    }
    
    # Save each trajectory's predictions
    for i in range(n_train_traj):
        save_dict[f'train_traj_{i}_Gamma_n'] = train_predictions['Gamma_n'][i]
        save_dict[f'train_traj_{i}_Gamma_c'] = train_predictions['Gamma_c'][i]
        save_dict[f'train_traj_{i}_X_OpInf'] = train_predictions['X_OpInf'][i]
    
    for i in range(n_test_traj):
        save_dict[f'test_traj_{i}_Gamma_n'] = test_predictions['Gamma_n'][i]
        save_dict[f'test_traj_{i}_Gamma_c'] = test_predictions['Gamma_c'][i]
        save_dict[f'test_traj_{i}_X_OpInf'] = test_predictions['X_OpInf'][i]
    
    np.savez(
        output_path + f"ensemble_trajectory_predictions_r{r}_k{len(best_models)}.npz",
        **save_dict
    )
    
    bprint(f"Saved to: {output_path}ensemble_trajectory_predictions_r{r}_k{len(best_models)}.npz")
    
else:
    bprint("WARNING: No valid models available for ensemble predictions!")

# %%
### Load the ensemble trajectory predictions
bprint("Loading ensemble trajectory predictions...")

# Determine number of models to load
if step_2 and best_models is not None:
    k = len(best_models)
else:
    k = num_models

pred_file = np.load(output_path + f"ensemble_trajectory_predictions_r{r}_k{k}.npz", allow_pickle=True)

n_train_traj = int(pred_file['n_train_traj'])
n_test_traj = int(pred_file['n_test_traj'])
num_models_used = int(pred_file['num_models_used'])
train_boundaries = pred_file['train_boundaries']
test_boundaries = pred_file['test_boundaries']

print(f"\nLoaded ensemble predictions:")
print(f"  Number of models in ensemble: {num_models_used}")
print(f"  Number of training trajectories: {n_train_traj}")
print(f"  Number of test trajectories: {n_test_traj}")

# Load predictions for each trajectory
train_predictions = {'Gamma_n': [], 'Gamma_c': [], 'X_OpInf': []}
test_predictions = {'Gamma_n': [], 'Gamma_c': [], 'X_OpInf': []}

for i in range(n_train_traj):
    train_predictions['Gamma_n'].append(pred_file[f'train_traj_{i}_Gamma_n'])
    train_predictions['Gamma_c'].append(pred_file[f'train_traj_{i}_Gamma_c'])
    train_predictions['X_OpInf'].append(pred_file[f'train_traj_{i}_X_OpInf'])
    print(f"  Train traj {i}: Gamma_n shape {train_predictions['Gamma_n'][i].shape}")

for i in range(n_test_traj):
    test_predictions['Gamma_n'].append(pred_file[f'test_traj_{i}_Gamma_n'])
    test_predictions['Gamma_c'].append(pred_file[f'test_traj_{i}_Gamma_c'])
    test_predictions['X_OpInf'].append(pred_file[f'test_traj_{i}_X_OpInf'])
    print(f"  Test traj {i}: Gamma_n shape {test_predictions['Gamma_n'][i].shape}")

# Load ground truth
bprint("\nLoading ground truth trajectories...")

train_truth = {'Gamma_n': [], 'Gamma_c': []}
for i, file_path in enumerate(training_files):
    fh = xr.open_dataset(file_path, engine=ENGINE, phony_dims="sort")
    train_truth['Gamma_n'].append(fh["gamma_n"].data)
    train_truth['Gamma_c'].append(fh["gamma_c"].data)
    fh.close()
    print(f"  Train truth {i}: {train_truth['Gamma_n'][i].shape}")

test_truth = {'Gamma_n': [], 'Gamma_c': []}
for i, file_path in enumerate(test_files):
    fh = xr.open_dataset(file_path, engine=ENGINE, phony_dims="sort")
    test_truth['Gamma_n'].append(fh["gamma_n"].data)
    test_truth['Gamma_c'].append(fh["gamma_c"].data)
    fh.close()
    print(f"  Test truth {i}: {test_truth['Gamma_n'][i].shape}")

bprint("Data loading complete!")

# %%
### Plot ensemble predictions vs ground truth for each trajectory
print("Creating ensemble prediction plots...")

# Compute ensemble statistics for each trajectory
def compute_ensemble_stats(predictions_list):
    """Compute mean and std across models for a list of trajectory predictions"""
    means = []
    stds = []
    for traj_preds in predictions_list:
        # traj_preds shape: (n_models, n_timesteps)
        means.append(np.mean(traj_preds, axis=0))
        stds.append(np.std(traj_preds, axis=0))
    return means, stds

train_Gamma_n_mean, train_Gamma_n_std = compute_ensemble_stats(train_predictions['Gamma_n'])
train_Gamma_c_mean, train_Gamma_c_std = compute_ensemble_stats(train_predictions['Gamma_c'])
test_Gamma_n_mean, test_Gamma_n_std = compute_ensemble_stats(test_predictions['Gamma_n'])
test_Gamma_c_mean, test_Gamma_c_std = compute_ensemble_stats(test_predictions['Gamma_c'])

###########################
# PLOT 1: Training Trajectories
###########################
fig, axes = plt.subplots(n_train_traj, 2, figsize=(16, 4*n_train_traj))
if n_train_traj == 1:
    axes = axes.reshape(1, -1)

for traj_idx in range(n_train_traj):
    time_steps = np.arange(len(train_Gamma_n_mean[traj_idx]))
    
    # Gamma_n
    ax = axes[traj_idx, 0]
    
    # Ensemble mean
    ax.plot(time_steps, train_Gamma_n_mean[traj_idx], 'b-', linewidth=2, label='Ensemble mean')
    
    # Uncertainty band
    ax.fill_between(time_steps,
                    train_Gamma_n_mean[traj_idx] - train_Gamma_n_std[traj_idx],
                    train_Gamma_n_mean[traj_idx] + train_Gamma_n_std[traj_idx],
                    alpha=0.3, color='blue', label='±1 std')
    
    # Ground truth
    ax.plot(time_steps, train_truth['Gamma_n'][traj_idx], 'k-', linewidth=1.5, 
            label='Ground truth', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Gamma_n', fontsize=11)
    ax.set_title(f'Training Trajectory {traj_idx + 1}: Gamma_n', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    
    # Gamma_c
    ax = axes[traj_idx, 1]
    
    # Ensemble mean
    ax.plot(time_steps, train_Gamma_c_mean[traj_idx], 'r-', linewidth=2, label='Ensemble mean')
    
    # Uncertainty band
    ax.fill_between(time_steps,
                    train_Gamma_c_mean[traj_idx] - train_Gamma_c_std[traj_idx],
                    train_Gamma_c_mean[traj_idx] + train_Gamma_c_std[traj_idx],
                    alpha=0.3, color='red', label='±1 std')
    
    # Ground truth
    ax.plot(time_steps, train_truth['Gamma_c'][traj_idx], 'k-', linewidth=1.5, 
            label='Ground truth', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Gamma_c', fontsize=11)
    ax.set_title(f'Training Trajectory {traj_idx + 1}: Gamma_c', fontsize=12)
    ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(output_path + f"ensemble_train_trajectories_r{r}_k{num_models_used}.png", 
            dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}ensemble_train_trajectories_r{r}_k{num_models_used}.png")
plt.show()

###########################
# PLOT 2: Test Trajectories
###########################
fig, axes = plt.subplots(n_test_traj, 2, figsize=(16, 4*n_test_traj))
if n_test_traj == 1:
    axes = axes.reshape(1, -1)

for traj_idx in range(n_test_traj):
    time_steps = np.arange(len(test_Gamma_n_mean[traj_idx]))
    
    # Gamma_n
    ax = axes[traj_idx, 0]
    
    # Ensemble mean
    ax.plot(time_steps, test_Gamma_n_mean[traj_idx], 'b-', linewidth=2, label='Ensemble mean')
    
    # Uncertainty band
    ax.fill_between(time_steps,
                    test_Gamma_n_mean[traj_idx] - test_Gamma_n_std[traj_idx],
                    test_Gamma_n_mean[traj_idx] + test_Gamma_n_std[traj_idx],
                    alpha=0.3, color='blue', label='±1 std')
    
    # Ground truth
    ax.plot(time_steps, test_truth['Gamma_n'][traj_idx], 'k-', linewidth=1.5, 
            label='Ground truth', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Gamma_n', fontsize=11)
    ax.set_title(f'Test Trajectory {traj_idx + 1}: Gamma_n', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    
    # Gamma_c
    ax = axes[traj_idx, 1]
    
    # Ensemble mean
    ax.plot(time_steps, test_Gamma_c_mean[traj_idx], 'r-', linewidth=2, label='Ensemble mean')
    
    # Uncertainty band
    ax.fill_between(time_steps,
                    test_Gamma_c_mean[traj_idx] - test_Gamma_c_std[traj_idx],
                    test_Gamma_c_mean[traj_idx] + test_Gamma_c_std[traj_idx],
                    alpha=0.3, color='red', label='±1 std')
    
    # Ground truth
    ax.plot(time_steps, test_truth['Gamma_c'][traj_idx], 'k-', linewidth=1.5, 
            label='Ground truth', alpha=0.7)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Gamma_c', fontsize=11)
    ax.set_title(f'Test Trajectory {traj_idx + 1}: Gamma_c', fontsize=12)
    ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(output_path + f"ensemble_test_trajectories_r{r}_k{num_models_used}.png", 
            dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}ensemble_test_trajectories_r{r}_k{num_models_used}.png")
plt.show()

# %%
### Compute quantitative error metrics
bprint("Computing error metrics...")

def compute_errors(pred_mean_list, truth_list, name=""):
    """Compute MAE and relative errors for a set of trajectories"""
    maes = []
    rel_errors = []
    
    for i, (pred, truth) in enumerate(zip(pred_mean_list, truth_list)):
        # Ensure same length
        min_len = min(len(pred), len(truth))
        pred = pred[:min_len]
        truth = truth[:min_len]
        
        mae = np.mean(np.abs(pred - truth))
        rel_error = mae / (np.mean(np.abs(truth)) + 1e-10)
        
        maes.append(mae)
        rel_errors.append(rel_error)
        
        print(f"  {name} Traj {i+1}: MAE={mae:.6f}, Rel Error={rel_error:.4%}")
    
    print(f"\n  {name} Average: MAE={np.mean(maes):.6f}, Rel Error={np.mean(rel_errors):.4%}")
    return maes, rel_errors

print("\n" + "="*60)
print("TRAINING RECONSTRUCTION ERRORS")
print("="*60)
print("\nGamma_n:")
train_Gamma_n_maes, train_Gamma_n_rels = compute_errors(
    train_Gamma_n_mean, train_truth['Gamma_n'], "Train Gamma_n"
)

print("\nGamma_c:")
train_Gamma_c_maes, train_Gamma_c_rels = compute_errors(
    train_Gamma_c_mean, train_truth['Gamma_c'], "Train Gamma_c"
)

print("\n" + "="*60)
print("TEST PREDICTION ERRORS (UNSEEN ICs)")
print("="*60)
print("\nGamma_n:")
test_Gamma_n_maes, test_Gamma_n_rels = compute_errors(
    test_Gamma_n_mean, test_truth['Gamma_n'], "Test Gamma_n"
)

print("\nGamma_c:")
test_Gamma_c_maes, test_Gamma_c_rels = compute_errors(
    test_Gamma_c_mean, test_truth['Gamma_c'], "Test Gamma_c"
)

# Compute average uncertainty
print("\n" + "="*60)
print("ENSEMBLE UNCERTAINTY (Avg Std Dev)")
print("="*60)
for i in range(n_train_traj):
    print(f"  Train Traj {i+1}: Gamma_n std={np.mean(train_Gamma_n_std[i]):.6f}, "
          f"Gamma_c std={np.mean(train_Gamma_c_std[i]):.6f}")

for i in range(n_test_traj):
    print(f"  Test Traj {i+1}: Gamma_n std={np.mean(test_Gamma_n_std[i]):.6f}, "
          f"Gamma_c std={np.mean(test_Gamma_c_std[i]):.6f}")

print("="*60 + "\n")

# Save metrics
np.savez(
    output_path + f"ensemble_metrics_r{r}_k{num_models_used}.npz",
    train_Gamma_n_maes=train_Gamma_n_maes,
    train_Gamma_n_rels=train_Gamma_n_rels,
    train_Gamma_c_maes=train_Gamma_c_maes,
    train_Gamma_c_rels=train_Gamma_c_rels,
    test_Gamma_n_maes=test_Gamma_n_maes,
    test_Gamma_n_rels=test_Gamma_n_rels,
    test_Gamma_c_maes=test_Gamma_c_maes,
    test_Gamma_c_rels=test_Gamma_c_rels
)
print(f"Metrics saved to: {output_path}ensemble_metrics_r{r}_k{num_models_used}.npz")


