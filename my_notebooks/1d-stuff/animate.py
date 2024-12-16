import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = jnp.load("test_lj.npy")[..., 0]

num_particles = data.shape[1]


# function to get max count for a given number of bins
def get_bin_count(data, bins):
    count, _ = jnp.histogram(data, bins)
    return count


get_bin_count_vec = jax.vmap(get_bin_count, in_axes=(0, None))

# animation
bins = 50
max_bins = jnp.max(get_bin_count_vec(data, bins))

hist = True
scatter = False

if hist:

    fig, ax = plt.subplots()

    def init():
        ax.cla()
        ax.set_xlim([0, 512])
        ax.set_ylim([0, max_bins])
        ax.hist(data[0, :], bins)

    def animate(i):
        ax.cla()
        ax.set_xlim([0, 512])
        ax.set_ylim([0, max_bins])
        ax.hist(data[i, :], bins)

    ani = FuncAnimation(
        fig,
        animate,
        frames=50,
        init_func=init,
        interval=10,
        blit=False,
    )

    plt.show()

if scatter:

    fig, ax = plt.subplots()

    ts = jnp.linspace(0, 1, 1000)

    def init():
        ax.cla()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 512])
        ax.scatter([], [])

    def animate(i):
        ax.cla()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 512])
        ax.scatter(jnp.tile(ts, (num_particles, 1)).T[i, :], data[i, :])

    ani = FuncAnimation(
        fig,
        animate,
        frames=1000,
        init_func=init,
        interval=10,
        blit=False,
    )

    plt.show()
