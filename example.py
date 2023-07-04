import jax.numpy as jnp
import jax
from ipddp import iplqr
from jax.config import config
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update("jax_platform_name", "cpu")

state_dim = 2
control_dim = 1

P = jnp.array([[3.04854389, -2.50545756], [-2.50545756, 12.99156241]])
Q = jnp.eye(state_dim)
R = 0.01 * jnp.eye(control_dim)

A = jnp.array([[0.7326, -0.0861], [0.1722, 0.9909]])
B = jnp.array([[0.0609], [0.0064]])

G = -jnp.eye(state_dim)
H = jnp.array([[1], [-1]])
g = jnp.array([0.5, 0.5])
h = jnp.array([2, 2])


horizon = 4
mean = jnp.array([0.0])
sigma = jnp.array([0.1])
key = jax.random.PRNGKey(1)
u = mean + sigma * jax.random.normal(key, shape=(horizon, control_dim))
x0 = jnp.array([1.0, 1.0])
xd = jnp.array([0.0, 0.0])
z = 1e-4 * jnp.ones((horizon, g.shape[0] + h.shape[0]))
s = 1e-4 * jnp.ones((horizon, g.shape[0] + h.shape[0]))


def mpc_body(carry, inp):
    prev_state, prev_control = carry
    states, control, _, _ = iplqr(
        prev_state,
        prev_control,
        xd,
        s,
        z,
        A,
        B,
        G,
        g,
        H,
        h,
        P,
        Q,
        R,
    )
    next_state = A @ prev_state + B @ control[0]
    return (next_state, control), (next_state, control[0])


_, mpc_out = jax.lax.scan(mpc_body, (x0, u), None, length=40)
mpc_states, mpc_controls = mpc_out
mpc_states = jnp.vstack((x0, mpc_states))

plt.plot(mpc_controls)
plt.show()
plt.plot(mpc_states[:, 0])
plt.plot(mpc_states[:, 1])
plt.show()
