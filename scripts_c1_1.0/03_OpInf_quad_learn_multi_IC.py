from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *

from config.HW import *

import xarray as xr

r = 78

ridge_alf_lin_all = np.linspace(1e2, 1e4, 10)
ridge_alf_quad_all = np.linspace(1e11, 1e14, 10)

gamma_reg_lin = np.linspace(1e-4, 1e1, 20)
gamma_reg_quad = np.linspace(1e-3, 1e2, 20)

save_dir = "/home/anthonypoole/output/IEEE/opinf_results/"
path_to_data = save_dir

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


###########################
print("\033[1m Prepare the data for learning (MULTI-IC version) \033[0m")

# Load combined training data from multiple initial conditions
Xhat_train_file = path_to_data + "X_hat_train_multi_IC.npy"
Xhatmax = np.load(Xhat_train_file)
Xhat = Xhatmax[:, :r]

print(f"Training data shape: {Xhat.shape}")

# Prepare state evolution data
X_state = Xhat[:-1, :]
Y_state = Xhat[1:, :]

s = int(r * (r + 1) / 2)
d_state = r + s
d_out = r + s + 1

X_state2 = get_x_sq(X_state)
D_state = np.concatenate((X_state, X_state2), axis=1)
D_state_2 = D_state.T @ D_state
print("\033[1m State learning data prepared \033[0m")

###########################
print("\033[1m Prepare the output learning data \033[0m")

X_out = Xhatmax[:, :r]
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
print("\033[1m Done \033[0m")

###########################
print("\033[1m Load derived quantities from all training trajectories \033[0m")

ENGINE = "h5netcdf"

training_files = [
    "/storage1/HW/paper/1.0_300_training_IC1.h5",
    "/storage1/HW/paper/1.0_300_training_IC2.h5",
    "/storage1/HW/paper/1.0_300_training_IC3.h5",
    "/storage1/HW/paper/1.0_300_training_IC4.h5",
    "/storage1/HW/paper/1.0_300_training_IC5.h5",
]

Gamma_n_list = []
Gamma_c_list = []

for file_path in training_files:
    fh = xr.open_dataset(file_path, engine=ENGINE)
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

# Check for NaNs in gamma data
if np.any(np.isnan(Gamma_n)):
    print("\033[91m WARNING: NaNs detected in Gamma_n training data! \033[0m")
if np.any(np.isnan(Gamma_c)):
    print("\033[91m WARNING: NaNs detected in Gamma_c training data! \033[0m")
if np.any(np.isinf(Gamma_n)):
    print("\033[91m WARNING: Infs detected in Gamma_n training data! \033[0m")
if np.any(np.isinf(Gamma_c)):
    print("\033[91m WARNING: Infs detected in Gamma_c training data! \033[0m")

print("\033[1m Done \033[0m")

###########################
print("\033[1m Load test initial condition \033[0m")

IC_data = np.load(path_to_data + "initial_conditions_multi_IC.npz")
test_IC_reduced = IC_data["test_IC_reduced"][:r]

print(f"Test IC shape: {test_IC_reduced.shape}")

###########################
print("\033[1m BEGIN ENSEMBLE LEARNING \033[0m")

prec_mean = 0.20
prec_std = 0.50

# For testing: number of steps to predict on test IC
n_steps_test = 5000

Gamma_n_ensemble = []
Gamma_c_ensemble = []

alphas_lin = []
alphas_quad = []

best_models = []  # Store best models for later analysis

for alpha_state_lin in ridge_alf_lin_all:
    for alpha_state_quad in ridge_alf_quad_all:
        print("alpha_lin = %.2E" % alpha_state_lin)
        print("alpha_quad = %.2E" % alpha_state_quad)

        regg = np.zeros(d_state)
        regg[:r] = alpha_state_lin
        regg[r : r + s] = alpha_state_quad
        regularizer = np.diag(regg)
        D_state_reg = D_state_2 + regularizer

        O = np.linalg.solve(D_state_reg, np.dot(D_state.T, Y_state)).T

        A = O[:, :r]
        F = O[:, r : r + s]
        f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))

        # Test on the test IC
        is_nan, Xhat_test_pred = solve_opinf_difference_model(test_IC_reduced, n_steps_test, f)

        if is_nan:
            print("  Skipping due to NaN in test prediction")
            continue

        X_test_pred = Xhat_test_pred.T
        Xhat_test_pred_scaled = (X_test_pred - mean_Xhat[np.newaxis, :]) / scaling_Xhat
        Xhat_2_test_pred_scaled = get_x_sq(Xhat_test_pred_scaled)

        # Check for NaNs in scaled test predictions
        if np.any(np.isnan(Xhat_test_pred_scaled)):
            print("  Skipping: NaN in scaled test predictions")
            continue
        if np.any(np.isinf(Xhat_test_pred_scaled)):
            print("  Skipping: Inf in scaled test predictions")
            continue

        # Learn output operators (on training data)
        for n, alpha_out_lin in enumerate(gamma_reg_lin):
            for m, alpha_out_quad in enumerate(gamma_reg_quad):
                print(
                    "\033[1m   Output reg params: {:.2E}, {:.2E} \033[0m".format(
                        alpha_out_lin, alpha_out_quad
                    )
                )

                regg = np.zeros(d_out)
                regg[:r] = alpha_out_lin
                regg[r : r + s] = alpha_out_quad
                regg[r + s :] = alpha_out_lin
                regularizer = np.diag(regg)
                D_out_reg = D_out_2 + regularizer

                # Check condition number
                cond_num = np.linalg.cond(D_out_reg)
                if cond_num > 1e15:
                    print(f"    Skipping: Poor conditioning (cond={cond_num:.2e})")
                    continue

                O = np.linalg.solve(D_out_reg, np.dot(D_out.T, Y_Gamma.T)).T

                # Check for NaNs in output operators
                if np.any(np.isnan(O)):
                    print("    Skipping: NaN in output operators O")
                    continue
                if np.any(np.isinf(O)):
                    print("    Skipping: Inf in output operators O")
                    continue

                C = O[:, :r]
                G = O[:, r : r + s]
                c = O[:, r + s]

                # Predict outputs on test trajectory
                Y_test_pred = (
                    C @ Xhat_test_pred_scaled.T
                    + G @ Xhat_2_test_pred_scaled.T
                    + c[:, np.newaxis]
                )

                # Check for NaNs in predictions
                if np.any(np.isnan(Y_test_pred)):
                    print("    Skipping: NaN in Y_test_pred")
                    continue
                if np.any(np.isinf(Y_test_pred)):
                    print("    Skipping: Inf in Y_test_pred")
                    continue

                ts_Gamma_n_test = Y_test_pred[0, :]
                ts_Gamma_c_test = Y_test_pred[1, :]

                # Compute statistics on test prediction
                mean_Gamma_n_test = np.mean(ts_Gamma_n_test)
                std_Gamma_n_test = np.std(ts_Gamma_n_test, ddof=1)

                mean_Gamma_c_test = np.mean(ts_Gamma_c_test)
                std_Gamma_c_test = np.std(ts_Gamma_c_test, ddof=1)

                # Compute errors relative to training statistics
                mean_err_Gamma_n = np.abs(mean_Gamma_n_ref - mean_Gamma_n_test) / mean_Gamma_n_ref
                std_err_Gamma_n = np.abs(std_Gamma_n_ref - std_Gamma_n_test) / std_Gamma_n_ref

                mean_err_Gamma_c = np.abs(mean_Gamma_c_ref - mean_Gamma_c_test) / mean_Gamma_c_ref
                std_err_Gamma_c = np.abs(std_Gamma_c_ref - std_Gamma_c_test) / std_Gamma_c_ref

                # Check if model meets accuracy criteria
                if (
                    mean_err_Gamma_n < prec_mean
                    and std_err_Gamma_n < prec_std
                    and mean_err_Gamma_c < prec_mean
                    and std_err_Gamma_c < prec_std
                ):
                    alphas_lin_temp = [alpha_state_lin, alpha_out_lin]
                    alphas_quad_temp = [alpha_state_quad, alpha_out_quad]

                    alphas_lin.append(alphas_lin_temp)
                    alphas_quad.append(alphas_quad_temp)

                    Gamma_n_ensemble.append(ts_Gamma_n_test)
                    Gamma_c_ensemble.append(ts_Gamma_c_test)

                    # Save model operators
                    best_models.append({
                        'A': A.copy(), 'F': F.copy(),
                        'C': C.copy(), 'G': G.copy(), 'c': c.copy(),
                        'alphas': [alpha_state_lin, alpha_state_quad, alpha_out_lin, alpha_out_quad]
                    })

                    print(f"    ✓ Model accepted! (Total: {len(Gamma_n_ensemble)})")

                print(
                    "    Test errors - Gamma_n: mean={:.3f}, std={:.3f} | Gamma_c: mean={:.3f}, std={:.3f}".format(
                        mean_err_Gamma_n, std_err_Gamma_n, mean_err_Gamma_c, std_err_Gamma_c
                    )
                )

print("\033[1m Done \033[0m")

###########################
print(f"\n\033[1m Found {len(Gamma_n_ensemble)} models meeting criteria \033[0m")

if len(Gamma_n_ensemble) > 0:
    Gamma_n_ensemble = np.asarray(Gamma_n_ensemble)
    Gamma_c_ensemble = np.asarray(Gamma_c_ensemble)

    Gamma_n_mean = np.mean(Gamma_n_ensemble, axis=0)
    Gamma_n_std = np.std(Gamma_n_ensemble, ddof=1, axis=0)

    Gamma_c_mean = np.mean(Gamma_c_ensemble, axis=0)
    Gamma_c_std = np.std(Gamma_c_ensemble, ddof=1, axis=0)

    print(f"Ensemble shapes: {Gamma_n_ensemble.shape}, {Gamma_c_ensemble.shape}")

    # Save results
    np.savez(
        "results/Gamma_ensemble_multi_IC_test_r" + str(r) + ".npz",
        Gamma_n_ensemble=Gamma_n_ensemble,
        Gamma_c_ensemble=Gamma_c_ensemble,
        Gamma_n_mean=Gamma_n_mean,
        Gamma_n_std=Gamma_n_std,
        Gamma_c_mean=Gamma_c_mean,
        Gamma_c_std=Gamma_c_std,
    )

    np.savez(
        "results/reg_params_ensemble_multi_IC_r" + str(r) + ".npz",
        alphas_lin=alphas_lin,
        alphas_quad=alphas_quad,
    )

    # Save best models
    np.savez(
        "results/best_models_multi_IC_r" + str(r) + ".npz",
        models=best_models
    )

    print("\033[1m Results saved! \033[0m")
else:
    print("\033[1m WARNING: No models met the accuracy criteria! \033[0m")
    print("Consider relaxing prec_mean and prec_std thresholds.")
