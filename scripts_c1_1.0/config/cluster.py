# Cluster settings
from .HW import *

data_dir = "/work2/10407/anthony50102/frontera/data/hw2d_sim/t600_d256x256_raw/" # Raw data directory
output_path = "/work2/10407/anthony50102/frontera/data/sciml_roms_hasegawa_wakatani/" # Output directory
training_files = [
    data_dir + "hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250315142044_11702_0.h5"
]
test_files = [
    data_dir + "hw2d_sim_step0.025_end1_pts512_c11_k015_N3_nu5e-8_20250316085602_1116_2.h5"
]

POD_file = output_path + "POD.npz"
Xhat_file = output_path + "X_hat.npy"

ridge_alf_lin_all = np.linspace(1e8, 1e4, 4)
ridge_alf_quad_all = np.linspace(1e20, 1e14, 6)

gamma_reg_lin = np.linspace(1e-4, 1e1, 4)
gamma_reg_quad = np.linspace(1e-3, 1e2, 5)
