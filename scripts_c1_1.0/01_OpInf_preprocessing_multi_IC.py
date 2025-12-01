from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *

import xarray as xr

from config.HW import *

if __name__ == "__main__":
    print("\033[1m Reading snapshots from multiple initial conditions...\033[0m")

    ENGINE = "h5netcdf"

    data_dir = "/home/anthonypoole/output/IEEE/test_trajs/c1_variations/"
    save_dir = "/home/anthonypoole/output/IEEE/opinf_results/"

    # List of training files - adjust paths to your data
    training_files = [
        data_dir + "hw2d_ss.025_time1000_grid512_c11.5_k0.15_N3_nu5e-8_down8.h5",
        data_dir + "hw2d_ss.025_time1000_grid512_c11_k0.15_N3_nu5e-8_down8.h5",
    ]

    # Test file (6th initial condition)
    test_file = data_dir + "hw2d_ss.025_time1000_grid512_c12_k0.15_N3_nu5e-8_down8.h5"


    Q_train_list = []
    
    for i, file_path in enumerate(training_files):
        print(f"  Loading IC {i+1}: {file_path}")
        fh = xr.open_dataset(file_path, engine=ENGINE)
        
        # TODO: These simulations don't include explicit x,y spatial coordinates in the dataset.
        # We infer the spatial grid structure from the shape of the density/phi arrays.
        # For these files, the grid is 64x64, so we reshape accordingly.
        
        # Get density and phi as numpy arrays
        density = fh["density"].values  # Shape should be (time, 64*64) or (time, 64, 64)
        phi = fh["phi"].values
        
        # If already 3D (time, y, x), use as-is. If 2D (time, flattened), reshape to 3D
        if density.ndim == 2:
            # Assume flattened: (time, 64*64) -> (time, 64, 64)
            n_time = density.shape[0]
            grid_size = 64  # TODO: Could also infer as int(np.sqrt(density.shape[1]))
            density = density.reshape(n_time, grid_size, grid_size)
            phi = phi.reshape(n_time, grid_size, grid_size)
        
        # Stack fields: (time, y, x) -> (field, time, y, x) -> flatten to (field*y*x, time)
        Q_ic = np.stack([density, phi], axis=0)  # Shape: (2, time, y, x)
        n_field, n_time, n_y, n_x = Q_ic.shape
        Q_ic = Q_ic.reshape(n_field * n_y * n_x, n_time)  # Shape: (2*64*64, time)
        
        Q_train_list.append(Q_ic)
        print(f"    Shape: {Q_ic.shape}")

    # Concatenate all training trajectories along time axis
    Q_train = np.hstack(Q_train_list)
    print(f"\n\033[1m Combined training data shape: {Q_train.shape}\033[0m")

    # Load test data
    print(f"\n\033[1m Loading test IC: {test_file}\033[0m")
    fh_test = xr.open_dataset(test_file, engine=ENGINE)
    
    # TODO: Same spatial grid inference as above
    density_test = fh_test["density"].values
    phi_test = fh_test["phi"].values
    
    if density_test.ndim == 2:
        n_time = density_test.shape[0]
        grid_size = 64  # TODO: Match with training data grid size
        density_test = density_test.reshape(n_time, grid_size, grid_size)
        phi_test = phi_test.reshape(n_time, grid_size, grid_size)
    
    Q_test = np.stack([density_test, phi_test], axis=0)
    n_field, n_time, n_y, n_x = Q_test.shape
    Q_test = Q_test.reshape(n_field * n_y * n_x, n_time)
    
    print(f"Test data shape: {Q_test.shape}")

    # Compute POD basis from combined training data
    print("\n\033[1m Computing POD basis from all training trajectories...\033[0m")
    U, S, _ = np.linalg.svd(Q_train, full_matrices=False)
    
    # Save POD data
    POD_file_multi = save_dir + "POD_multi_IC.npz"
    np.savez(POD_file_multi, S=S, Vr=U[:, :svd_save])
    print(f"  Saved POD basis to {POD_file_multi}")

    # Project training data
    print("\n\033[1m Projecting training data...\033[0m")
    Xhat_train = Q_train.T @ U
    Xhat_train_file = save_dir + "X_hat_train_multi_IC.npy"
    np.save(Xhat_train_file, Xhat_train)
    print(f"  Saved to {Xhat_train_file}")

    # Project test data
    print("\033[1m Projecting test data...\033[0m")
    Xhat_test = Q_test.T @ U
    Xhat_test_file = save_dir + "X_hat_test_multi_IC.npy"
    np.save(Xhat_test_file, Xhat_test)
    print(f"  Saved to {Xhat_test_file}")

    # Save initial conditions for later use
    print("\n\033[1m Saving initial conditions...\033[0m")
    train_ICs = np.array([Q_train_list[i][:, 0] for i in range(len(Q_train_list))])
    test_IC = Q_test[:, 0]
    
    np.savez(
        save_dir + "initial_conditions_multi_IC.npz",
        train_ICs=train_ICs,
        test_IC=test_IC,
        train_ICs_reduced=np.array([Xhat_train[i*Q_train_list[0].shape[1], :] 
                                     for i in range(len(Q_train_list))]),
        test_IC_reduced=Xhat_test[0, :]
    )

    print("\n\033[1m Done.\033[0m")
