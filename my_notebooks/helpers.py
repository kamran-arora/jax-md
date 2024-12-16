from jax_md import *
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt


def init_periodic_box(N, dim, avg_num_density):
    """
    Initialise a periodic box at given average number density

    Inputs
    ----------
    N: number of particles
    dim: dimension
    avg_num_density: average number density

    Returns
    ----------
    box_size: side length of periodic box
    displacement:
    shift:
    """
    box_size = quantity.box_size_at_number_density(N, avg_num_density, dim)
    displacement, shift = space.periodic(box_size)
    return box_size, displacement, shift


def init_free_space(N, dim):
    """
    Initialise N particles in free space

    Inputs
    ----------
    N: number of particles
    dim: dimension

    Returns
    ----------
    displacement:
    shift:
    """
    displacement, shift = space.free()
    return displacement, shift


def brownian_simulation(
    key, temperature, dt, num_steps, N, shift, energy_fn, init_cond
):
    """
    Simulates one realisation of overdamped Langevin dynamics

    Inputs
    ---------
    key: JAX.random PRNGKey
    temperature:
    dt: timestep
    num_steps: number of timesteps
    N: number of particles
    shift:
    energy_fn: specifies the potential governing the system
    init_cond: callable function that specifies the initial condition

    Returns
    -----------
    all_states: array of shape (num_steps+2, N, dim)
    """
    pos_key, sim_key = jr.split(key, 2)
    # R = jr.uniform(pos_key, (N, 2), minval=4.5, maxval=5.5)
    # R = jnp.zeros((N, 2))
    R = init_cond(key=pos_key, N=N)

    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
    initial_state = init_fn(sim_key, R)
    state = init_fn(sim_key, R)

    do_step = lambda state, t: (apply_fn(state), apply_fn(state))
    final_state, intermediate_states = jax.lax.scan(do_step, state, num_steps)

    initial = initial_state.position
    intermediate = intermediate_states.position
    final = final_state.position

    final = jnp.expand_dims(final, 0)
    initial = jnp.expand_dims(initial, 0)

    all_states = jnp.concatenate((initial, intermediate, final), axis=0)

    return all_states


"""
Define a vmapped version of brownian_simulation
- takes an array of keys as first argument
- returns array of shape (len(keys), num_steps+2, N, dim)
"""
brownian_simulation_vec = jax.vmap(
    brownian_simulation, in_axes=(0, None, None, None, None, None, None, None)
)


def rdf_pairwise(dists, dr, nbins):
    """
    Compute the RDF given a collection of pairwise distances between particles

    Inputs:
    ------------
    dists: Array, (N, N)
    dr: bin width
    nbins: number of bins

    Returns:
    ------------
    _: Array, (nbins,) of number of particles in each bin
    """
    # set zeros to length+1 so bincount ignores
    dists = jnp.where(jnp.abs(dists) < 1e-8, nbins + 1, dists)
    bins = jnp.floor(dists / dr).astype(int)
    return jnp.bincount(bins.flatten(), length=nbins)


def rdf_pairwise_times_and_sims(dists, dr, nbins):
    """
    Compute the RDF given a collection of pairwise distances between particles but batched over timesteps and simulations

    Inputs:
    ------------
    dists: Array, (S, M, N, N)
    dr: bin width
    nbins: number of bins

    Returns:
    ------------
    _: Array, (S, M, nbins)
    """
    return jax.vmap(jax.vmap(rdf_pairwise, (0, None, None)), (0, None, None))(
        dists, dr, nbins
    )
