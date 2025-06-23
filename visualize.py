import numpy as np
import matplotlib.pyplot as plt
from envs.learning_path_env import LearningPathEnv

def plot_q_values(Q, env):
    H, W = env.grid_shape
    fig, ax = plt.subplots()
    # heatmap of max-Q
    max_Q = Q.max(axis=1).reshape(H, W)
    im = ax.imshow(max_Q, cmap='viridis')
    plt.colorbar(im, ax=ax, label='max Q-value')
    # overlay policy arrows
    action_arrows = {0:(-0.3,0), 1:(0,0.3), 2:(0.3,0), 3:(0,-0.3)}
    for s in range(Q.shape[0]):
        r, c = divmod(s, W)
        best_a = np.argmax(Q[s])
        dx, dy = action_arrows[best_a]
        ax.arrow(c, r, dy, dx, head_width=0.1, length_includes_head=True)

    ax.set_title("Learned policy & Q-values")
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    # same env setup as train.py
    prereqs = { (1,1):(0,1), (2,2):(1,2) }
    env = LearningPathEnv(grid_shape=(4,4), start=(0,0), goal=(3,3),
                          prerequisites=prereqs)
    Q = np.load("q_table.npy")
    plot_q_values(Q, env)
