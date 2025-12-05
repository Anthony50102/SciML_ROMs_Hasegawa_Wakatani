import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as ani

argparse = argparse.ArgumentParser()
argparse.add_argument("--c", type=bool, default=False, help="Use cluster settings")
args = argparse.parse_args()

if args.c:
    print(f"\033[1m Using cluster settings \033[0m")
    from config.cluster import *
else:
    print(f"\033[1m Using local settings \033[0m")
    from config.local import *

# Load the full state prediction
train_recon_file = np.load(output_path + "training_reconstruction_multi_IC_r78.npz")
print(train_recon_file.keys())

X_full = train_recon_file["X_recon_full"].reshape(-1, 2, 256, 256)
print(X_full.shape)

fig, ax = plt.subplots()

X_full_sub = X_full[::10, 0, ...]

min_val = X_full_sub.min()
max_val = X_full_sub.max()
print(min_val, max_val)

img = plt.imshow(X_full_sub[0,...], vmin=min_val, vmax=max_val)

def animate(frame):
    img.set_data(X_full_sub[0, ...])
    return [img]

print(X_full_sub.shape)
print(X_full_sub.shape[0])
anim = ani.FuncAnimation(fig, animate, frames=int(X_full_sub.shape[0]), interval=50, blit=True)

anim.save('animation.gif', writer='pillow', fps=30)
