import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from cycler import cycler

try:
    data = jnp.load("blj.npy")
    print("data loaded successfully")
except:
    raise Exception("Error loading data")

print(jnp.shape(data))

N = jnp.shape(data)[1]
print(N)


def get_limits_2d(data):
    """
    assumes data is of shape (times, sample, coordinate)
    """
    xdata = data[:, :, 0]
    xdata_clean = jnp.nan_to_num(xdata)
    ydata = data[:, :, 1]
    ydata_clean = jnp.nan_to_num(ydata)

    xmax = jnp.max(xdata_clean)
    ymax = jnp.max(ydata_clean)
    xmin = jnp.min(xdata_clean)
    ymin = jnp.min(ydata_clean)

    if xmax >= 0:
        xmaxlim = 1.05 * xmax
    elif xmax < 0:
        xmaxlim = 0.95 * xmax
    if xmin >= 0:
        xminlim = 0.95 * xmin
    elif xmin < 0:
        xminlim = 1.05 * xmin
    if ymax >= 0:
        ymaxlim = 1.05 * ymax
    elif ymax < 0:
        ymaxlim = 0.95 * ymax
    if ymin >= 0:
        yminlim = 0.95 * ymin
    elif ymin < 0:
        yminlim = 1.05 * ymin
    return xmaxlim, xminlim, ymaxlim, yminlim


xM, xm, yM, ym = get_limits_2d(data)

# color cycler
custom_cycler = cycler("color", plt.cm.Spectral(jnp.linspace(0, 1, N)))

fig, ax = plt.subplots()


def init():
    ax.cla()
    ax.set_prop_cycle(custom_cycler)
    ax.set_xlim([xm, xM])
    ax.set_ylim([ym, yM])
    ax.plot([], [])


def animate(i):
    ax.cla()
    ax.set_prop_cycle(custom_cycler)
    ax.set_xlim([xm, xM])
    ax.set_ylim([ym, yM])
    # plot just the current position of the particles
    # ax.plot(
    #     data[(i * 10) - 1 : i * 10, :, 0], data[(i * 10) - 1 : i * 10, :, 1], marker="o"
    # )
    # plot the trajectories of the particles
    ax.scatter(data[i, :, 0], data[i, :, 1])
    # ax.plot(data[1 : i * 10, :, 0], data[1 : i * 10, :, 1])


ani = FuncAnimation(
    fig,
    animate,
    frames=jnp.shape(data)[0],
    init_func=init,
    interval=100,
    blit=False,
    repeat=False,
)

# writer = PillowWriter(fps=30)
# ani.save("test.gif", writer=writer)
plt.show()
