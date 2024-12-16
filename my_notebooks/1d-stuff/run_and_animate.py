import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from jax_md import *
from jax_md import energy
from matplotlib.animation import FuncAnimation


jax.config.update("jax_enable_x64", True)

"""
Simulate one-dimensional overdamped Langevin dynamics with periodic BCs using JAX-MD

1. define setup parameters e.g. number of particles, interval length etc.
2. define displacement/shift/metric on space
3. define a function to simulate the system
4. define a custom potential, promote it to act on pairs of particles, and define the associated energy function
5. run the simulation, and store the output at every timestep
6. visualisations
"""

# RNG
SEED = 123
key = jr.key(SEED)

# setup
dim = 1
N = 256
box_size = N * 1.5
density = N / box_size
print(f"Initial uniform density: {density:.2f}")
dt = 0.01
timesteps = 10000
# temperature - this is the diffusion constant in the continuum model
kT = 0.01

# displacement
displacement, shift = space.periodic(box_size)
metric = space.metric(displacement)
v_displacement = space.map_product(displacement)
v_metric = space.map_product(metric)


# brownian simulation
def brownian_simulation(key, temperature, dt, steps, energy_fn):
    pos_key, sim_key = jr.split(key, 2)
    # uniform initial condition
    R = jr.uniform(pos_key, (N, 1), maxval=box_size)

    # overdamped Langevin dynamics simulation
    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
    apply_fn = jax.jit(apply_fn)
    initial_state = init_fn(sim_key, R)
    state = init_fn(sim_key, R)

    # loop using lax
    do_step = lambda state, t: (apply_fn(state), apply_fn(state))
    final_state, intermediate_states = jax.lax.scan(do_step, state, steps)

    return (
        initial_state.position,
        final_state.position,
        intermediate_states.position,
    )


# define a custom potential
def custom_potential(dr, A=1.0, B=1.0, sig1=0.1, sig2=0.3, **kwargs):
    U = -A * jnp.exp(-(dr**2) / (2 * sig1**2)) + B * jnp.exp(-(dr**2) / (2 * sig2**2))
    return jnp.array(U, dtype=dr.dtype)


# promote to act on pairs of particles
def custom_potential_pair(
    displacement_or_metric, species=None, A=1.0, B=1.0, sig1=0.1, sig2=0.3
):
    A = jnp.array(A, dtype=jnp.float32)
    B = jnp.array(B, dtype=jnp.float32)
    sig1 = jnp.array(sig1, dtype=jnp.float32)
    sig2 = jnp.array(sig2, dtype=jnp.float32)
    return smap.pair(
        custom_potential,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        A=A,
        B=B,
        sig1=sig1,
        sig2=sig2,
    )


# energy function
energy_fn = custom_potential_pair(
    displacement, A=10.0, sig1=0.1 * box_size, sig2=0.3 * box_size
)

# run simulation
try:
    initial, final, intermediate = brownian_simulation(
        key,
        kT,
        dt,
        jnp.arange(int(timesteps)),
        energy_fn,
    )
    print("simulation succesful")
except:
    raise Exception("simulation failed")

# clean up
final = jnp.expand_dims(final, 0)
initial = jnp.expand_dims(initial, 0)
all_states = jnp.concatenate((initial, intermediate, final), axis=0)
print(f"output shape: {jnp.shape(all_states)}")

# extract data for animation
data = all_states[..., 0]


# function to get max count for a given number of bins
def get_bin_count(data, bins):
    count, _ = jnp.histogram(data, bins)
    return count


get_bin_count_vec = jax.vmap(get_bin_count, in_axes=(0, None))

# animation
bins = 100
max_bins = jnp.max(get_bin_count_vec(data, bins))

hist = True
scatter = False

# plot final particle distribution
plt.hist(data[-1, :])
plt.xlim([0, box_size])
plt.title("Distribution at final time")
plt.show()

if hist:

    fig, ax = plt.subplots()

    def init():
        ax.cla()
        ax.set_xlim([0, box_size])
        ax.set_ylim([0, max_bins])
        ax.hist(data[0, :], bins)
        ax.set_title("start")

    def animate(i):
        ax.cla()
        ax.set_xlim([0, box_size])
        ax.set_ylim([0, max_bins])
        ax.hist(data[i, :], bins, edgecolor="black")

    ani = FuncAnimation(
        fig,
        animate,
        frames=int(timesteps / 20),
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
        ax.set_ylim([0, box_size])
        ax.scatter([], [])

    def animate(i):
        ax.cla()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, box_size])
        ax.scatter(jnp.tile(ts, (N, 1)).T[i, :], data[i, :])

    ani = FuncAnimation(
        fig,
        animate,
        frames=1000,
        init_func=init,
        interval=10,
        blit=False,
    )

    plt.show()

# compute and plot RDF at final time
radii = jnp.linspace(0, box_size, 51)
dr = box_size / 51
g = quantity.pair_correlation(metric, radii, dr, compute_average=True)
rdf = g(all_states[-1, ...])
fig_rdf, ax_rdf = plt.subplots()
ax_rdf.plot(radii, rdf)
ax_rdf.set_title("RDF at final time")
plt.show()
