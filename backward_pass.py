import jax.numpy as jnp
import jax.lax
import jax.scipy as jcp


def bwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    goal_state: jnp.ndarray,
    nominal_slacks: jnp.ndarray,
    nominal_duals: jnp.ndarray,
    barrier_param: float,
    reg_param: float,
    A,
    B,
    G,
    g,
    H,
    h,
    P,
    Q,
    R,
):
    def bwd_step(carry, inp):
        # unpack carry
        Vx, Vxx = carry

        # state, controls, dual
        state, control, slack, dual = inp

        # compute derivatives at nominal_values
        cx = jnp.vstack((G, jnp.zeros((H.shape[0], G.shape[1]))))
        cu = jnp.vstack((jnp.zeros((G.shape[0], H.shape[1])), H))

        # Q function expansion coeffs
        Qx = Q @ (state - goal_state) + cx.T @ dual + A.T @ Vx
        Qu = R @ control + cu.T @ dual + B.T @ Vx
        Qxx = Q + A.T @ Vxx @ A
        Quu = R + B.T @ Vxx @ B
        Quu += reg_param * jnp.eye(control.shape[0])
        Qxu = A.T @ Vxx @ B
        Quu = (Quu + Quu.T) / 2.0

        # perturbed KKT
        Slack_inv = jnp.diag(1 / slack)
        Sigma = Slack_inv @ jnp.diag(dual)
        rp = jnp.hstack((G @ state - g, H @ control - h)) + slack
        rd = dual * slack - barrier_param
        r = dual * rp - rd
        KKT_mat = Quu + cu.T @ Sigma @ cu

        chol_and_lower = jcp.linalg.cho_factor(KKT_mat)
        k_control = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ r - Qu)
        K_control = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ cx - Qxu.T)
        k_dual = Sigma @ cu @ k_control + Slack_inv @ r
        K_dual = Sigma @ (cu @ K_control + cx)
        k_slack = -rp - cu @ k_control
        K_slack = -(cu @ K_control + cx)

        # update Q function expansion parameters
        Qx = Qx + cx.T @ Slack_inv @ r
        Qu = Qu + cu.T @ Slack_inv @ r
        Qxx = Qxx + cx.T @ Sigma @ cx
        Qxu = Qxu + cx.T @ Sigma @ cu
        Quu = Quu + cu.T @ Sigma @ cu

        # Value function expansion parameters and diff cost
        Vx = Qx + Qxu @ k_control
        Vxx = Qxx + Qxu @ K_control
        dV = k_control.T @ Qu + 0.5 * k_control.T @ Quu @ k_control.T

        error = jnp.hstack((Qu, rp, rd))
        return (Vx, Vxx), (
            k_control,
            k_dual,
            k_slack,
            K_control,
            K_dual,
            K_slack,
            dV,
            error,
        )

    xN = nominal_states[-1]
    VxN = P @ (xN - goal_state)
    VxxN = P

    carry_out, bwd_pass_out = jax.lax.scan(
        bwd_step,
        (VxN, VxxN),
        (nominal_states[:-1], nominal_controls, nominal_slacks, nominal_duals),
        reverse=True,
    )
    (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        optimality_error,
    ) = bwd_pass_out
    diff_cost = jnp.sum(diff_cost)

    return (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        jnp.max(jnp.abs(optimality_error)),
    )
