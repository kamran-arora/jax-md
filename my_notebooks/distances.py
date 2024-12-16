import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def cdist(x: Float[Array, "N d"], y: Float[Array, "N d"]) -> Float[Array, "N N"]:
    """
    Computes the pairwise euclidean distance between two collections of points

    Inputs:
    ------------
    x: Array (N, d)
    y: Array (N, d)

    Returns:
    ------------
    _: Array (N, N)
    """
    return jnp.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))


def pdist(x: Float[Array, "N d"]) -> Float[Array, "N N"]:
    """
    Compute pairwise euclidean distances between a collection of points

    Inputs:
    ------------
    x: Array (N, d)

    Returns:
    ------------
    _: Array (N, N)
    """
    return cdist(x, x)


def pdist_sims_and_times(x: Float[Array, "S M N d"]) -> Float[Array, "S M N N"]:
    """
    Computes pairwise euclidean distances between a collection of points but batched across timesteps and number of simulations

    Inputs:
    -------------
    x: Array (S, M, N, d)

    S: number of simulations,
    M: number of timesteps,
    N: number of points,
    d: dimension

    Returns:
    -------------
    _: Array (S, M, N, N)

    i.e. the (N, N) distance matrix at all times for all simulations
    """
    return jax.vmap(jax.vmap(pdist, 0), 0)(x)


################################################################################
################################################################################
################################################################################


def nearest_image_two_points(p1, p2, L):
    """
    Compute the euclidean nearest image between two points in a hypercube with PBCs

    Inputs:
    --------------
    p1: Array, (d,)
    p2: Array, (d,)
    L: Array of sidelengths, (d,)

    Returns:
    --------------
    _: nearest image distance
    """
    r = p1 - p2
    r = r - L * jnp.rint(r / L)
    return jnp.sqrt(jnp.sum(r * r))


def nearest_image(p, L):
    """
    Compute pairwise euclidean nearest images between a collection of points in a hypercube with PBCs

    Inputs:
    -------------
    p: Array, (N, d)
    L: Array of sidelengths, (d,)

    Returns:
    ---------------
    _: Array (N, N)
    """
    return jax.vmap(
        lambda p1: jax.vmap(lambda p2: nearest_image_two_points(p1, p2, L))(p)
    )(p)


def nearest_image_sims_and_times(p, L):
    """
    Computes pairwise euclidean nearest image distances between a collection of points but batched across timesteps and number of simulations

    Inputs:
    -------------
    p: Array (S, M, N, d)

    S: number of simulations,
    M: number of timesteps,
    N: number of points,
    d: dimension

    L: Array of sidelengths, (d,)

    Returns:
    -------------
    _: Array (S, M, N, N)

    i.e. the (N, N) distance matrix at all times for all simulations
    """
    return jax.vmap(jax.vmap(nearest_image, (0, None)), (0, None))(p, L)
