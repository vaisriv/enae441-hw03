import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power, inv
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# CONSTANTS
# Gravitational parameter for Earth in km^3/s^2
mu = 398600  # km^3/s^2
omega_EN = 7.2921150e-5  # rad /s

# Problem 2
X_0_spring = np.array([1, 0])  # m, m/s

X_N_0 = np.array([7000, 0, 0, 0, 7.5, 3.5])  # km, km/s
dX_N_0 = np.array([30, 0, 0, 0, 0, 0.1])  # km, km/s

cur_dir = os.path.dirname(__file__)

# Helpers
m = 1.0
c = 0.5
k = 4.0

A = np.array([[0.0, 1.0], [-k/m, -c/m]], dtype=float)
C = np.array([[1.0, 0.0]], dtype=float)
D = np.array([[0.0]], dtype=float)

def _x_dot(t, x):
    return (A @ x)

deg = np.pi/180.0
phi_lat = 30.0 * deg
lam_lon = 60.0 * deg
R_E = 6378.0
omega_E = 7.2921150e-5

def R3(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

# Observer ECEF at t=0 from lat/lon
r_obs_ecef0 = np.array([
    R_E*np.cos(phi_lat)*np.cos(lam_lon),
    R_E*np.cos(phi_lat)*np.sin(lam_lon),
    R_E*np.sin(phi_lat)
], dtype=float)

def observer_position_ECI(t):
    return R3(omega_E * float(t)) @ r_obs_ecef0

def kepler_f(t, x):
    r = x[0:3]
    v = x[3:6]
    rn = np.linalg.norm(r)
    a = -mu * r / (rn**3)
    return np.hstack([v, a])

def integrate_trajectory(x_0, t_vec):
    sol = solve_ivp(kepler_f, (float(t_vec[0]), float(t_vec[-1])), x_0,
                    t_eval=t_vec, rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T  # (len(t_vec), 6)

def integrate_nominal_and_stm(X_0, t_vec):
    n = 6
    Phi0 = np.eye(n)
    z0 = np.hstack([X_0, Phi0.flatten(order="F")])

    def z_dot(t, z):
        x = z[:n]
        Phi = z[n:].reshape((n, n), order="F")
        A = get_Ak(x)
        xdot = kepler_f(t, x)
        Phidot = A @ Phi
        return np.hstack([xdot, Phidot.flatten(order="F")])

    sol = solve_ivp(z_dot, (float(t_vec[0]), float(t_vec[-1])), z0,
                    t_eval=t_vec, rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)

    Z = sol.y.T
    X_nom = Z[:, :n]
    Phi_t = np.empty((len(t_vec), n, n))
    for i in range(len(t_vec)):
        Phi_t[i] = Z[i, n:].reshape((n, n), order="F")
    return X_nom, Phi_t

###############################################
# OPTIONAL FUNCTIONS TO AID IN DEBUGGING
# These are not graded functions, but may help you debug your code.
# Keep the function signatures the same if you want autograder feedback!
###############################################


def load_numpy_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    print(f"Data keys loaded from {file_path}: {list(data.keys())}")
    print(
        "Query the data dictionary using `data['key_name']` to access specific data arrays."
    )
    return data


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


#######################
# Problem 1
#######################


# REQUIRED --- 1a
def propogate_CT_LTI_numerically(X_0, X_dot_fcn, t_vec):
    t0 = float(t_vec[0])
    tf = float(t_vec[-1])

    reversed_time = tf < t0
    if reversed_time:
        t_eval = np.asarray(t_vec)[::-1]
        t_span = (float(t_eval[0]), float(t_eval[-1]))
    else:
        t_eval = np.asarray(t_vec)
        t_span = (t0, tf)

    sol = solve_ivp(
        fun=X_dot_fcn,
        t_span=t_span,
        y0=np.asarray(X_0).reshape(-1),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    X_t = sol.y.T
    if reversed_time:
        X_t = X_t[::-1]

    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1b
def propogate_CT_LTI_analytically(X_0, A, t_vec):
    X_0 = np.asarray(X_0).reshape(-1)
    A = np.asarray(A)
    t_vec = np.asarray(t_vec)
    t0 = float(t_vec[0])

    X_t = np.empty((len(t_vec), len(X_0)), dtype=float)
    for i, t in enumerate(t_vec):
        Phi = expm(A * (float(t) - t0))
        X_t[i] = Phi @ X_0

    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1c
def propogate_DT_LTI_analytically(X_0, A, dt, k_max):
    X_0 = np.asarray(X_0).reshape(-1)
    A = np.asarray(A)
    A_d = expm(A * float(dt))

    n = X_0.size
    X_t = np.empty((int(k_max) + 1, n), dtype=float)

    # x[0]
    X_t[0] = X_0

    # x[k] = A_d^k x[0]
    # (explicitly use matrix_power as the assignment requests)
    for k in range(1, int(k_max) + 1):
        A_dk = np.linalg.matrix_power(A_d, k)
        X_t[k] = A_dk @ X_0

    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1d
def propagate_LTV_system_numerically(X_0, x_dot_fcn, A_fcn, t_vec):
    n = len(X_0)
    phi0 = np.eye(n).flatten()
    z0 = np.hstack((X_0, phi0))

    def z_dot(t, z):
        x = z[:n]
        phi = z[n:].reshape((n, n))

        x_dot = x_dot_fcn(t, x)
        A_t = A_fcn(x)
        phi_dot = A_t @ phi

        return np.hstack((x_dot, phi_dot.flatten()))

    sol = solve_ivp(z_dot, [t_vec[0], t_vec[-1]], z0, t_eval=t_vec)

    X_t_vec = sol.y[:n, :].T
    phi_t_vec = np.zeros((len(t_vec), n, n))
    for i in range(len(t_vec)):
        phi_t_vec[i] = sol.y[n:, i].reshape((n, n))

    # Return trajectory and STM over time where
    # np.shape(X_t_vec) = (len(t_vec), len(X_0))
    # np.shape(phi_t_vec) = (len(t_vec), len(X_0), len(X_0))

    return X_t_vec, phi_t_vec


# REQUIRED --- 1e
def max_allowable_sampling_time(A):
    A = np.asarray(A, dtype=float)
    lam = np.linalg.eigvals(A)
    dt_max = np.pi / (2*np.max(np.abs(lam)))

    return dt_max


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2b
def plot_trajectories():
    t0, tf = 0.0, 10.0
    t_ct = np.linspace(t0, tf, 2001)

    # CT numerical
    sol = solve_ivp(_x_dot, (t_ct[0], t_ct[-1]), X_0_spring, t_eval=t_ct, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    X_ct_num = sol.y.T  # (N,2)

    # CT analytic
    X_ct_ana = np.empty_like(X_ct_num)
    t_ref = t_ct[0]
    for i, t in enumerate(t_ct):
        X_ct_ana[i] = expm(A * (t - t_ref)) @ X_0_spring

    # DT analytic
    dt_good = 0.02
    k_max = int(np.floor((tf - t0) / dt_good))
    t_dt = t0 + np.arange(k_max + 1) * dt_good
    A_d = expm(A * dt_good)

    X_dt = np.empty((k_max + 1, 2))
    X_dt[0] = X_0_spring
    for k in range(1, k_max + 1):
        X_dt[k] = np.linalg.matrix_power(A_d, k) @ X_0_spring

    # plot
    fig, ax = plt.subplots()
    ax.plot(t_ct, X_ct_num[:, 0], label="x (CT numeric)")
    ax.plot(t_ct, X_ct_num[:, 1], label="xdot (CT numeric)", linestyle="--")

    ax.plot(t_ct, X_ct_ana[:, 0], label="x (CT analytic)")
    ax.plot(t_ct, X_ct_ana[:, 1], label="xdot (CT analytic)", linestyle="--")

    ax.plot(t_dt, X_dt[:, 0], label="x (DT analytic)", marker="o", markevery=80, linestyle="none")
    ax.plot(t_dt, X_dt[:, 1], label="xdot (DT analytic)", marker="s", markevery=80, linestyle="none")

    ax.set_xlabel("t [s]")
    ax.set_ylabel("state")
    ax.set_title("Spring–Mass–Damper: CT vs DT propagation (10 s)")
    ax.grid(True)
    ax.legend(ncol=2)

    return fig


# REQUIRED --- Problem 2c
def describe_propagation_methods():
    return """
    Comparison:
    - CT numerical (solve_ivp) integrates \\dot{x} = A x and returns x(t) at requested t. Accuracy depends on tolerances;
      it handles stiff/ill-conditioned cases with adaptive steps.
    - CT analytic uses the continuous-time STM Φ(t, t0) = expm(A (t-t0)), giving the exact state at any t for LTI systems.
    - DT analytic samples the exact CT solution at multiples of Δt by forming A_d = expm(A Δt) and x[k] = A_d^k x[0].
      At the sample instants, this is mathematically identical to CT analytic.

    Critical sampling:
    Let λ = σ ± j ω_d be complex poles and ω_n = |λ| the undamped natural frequency. The Nyquist bound is
        Δt_max = π / max ω_n.
    For this system, max ω_n = 2 rad/s ⇒ Δt_max = π/2 ≈ 1.5708 s.

    What happens if Δt > Δt_max?
    - The discrete-time eigenangle is θ = ω_d Δt. Angles are wrapped modulo 2π in discrete time, producing an
      aliased apparent frequency ω_alias = |ω_d − 2π/Δt · round(ω_d Δt / 2π)|.
    - The DT trajectory x[k] = A_d^k x[0] is still the exact CT solution *at those sample instants*, but if you attempt to
      infer the underlying oscillation from samples (or connect samples with lines), you will see a lower, incorrect
      frequency (and possibly sign flips when θ ≈ π). Estimation and identification tasks will misinterpret the dynamics.
    - Stability is preserved here because |e^{λ Δt}| = e^{σ Δt} < 1 (σ < 0), but phase is misrepresented beyond Nyquist.
    """


# REQUIRED --- Problem 2d
def determine_x0():
    data = load_numpy_data(cur_dir + "/data/HW3-spring-data.npy")
    t_meas = np.asarray(data["t"], dtype=float).reshape(-1)
    y_meas = np.asarray(data["y"], dtype=float).reshape(-1)

    rows = [C]
    for i in range(1, len(t_meas)):
        phi = expm(A*(t_meas[i] - t_meas[i-1]))
        rows.append(C @ matrix_power(phi, i))
    Q = np.vstack(rows)
    Y = y_meas.reshape(-1, 1)

    X0 = inv(Q.T @ Q) @ Q.T @ Y

    return X0  # dimension (2,)


# REQUIRED --- Problem 2e
def describe_observability():
    # compute how many measurements you need and return value and description
    num_measurements = 2

    return f"""
        {num_measurements} measurements (e.g., at k=0 and k=1 with Δt=1 s) are sufficient.
        Reason: the 2×2 observability matrix O₂ = [ C ; C A_d ] equals
            [[1, 0],
             [Φ₁₁(Δt), Φ₁₂(Δt)]],
        which has full rank iff Φ₁₂(Δt) ≠ 0. For this system,
            Φ₁₂(Δt) = e^(-α Δt) · (1/ω_d) · sin(ω_d Δt),
        and with α=0.25, ω_d≈1.9843 rad/s, Δt=1 s ⇒ sin(ω_d Δt) ≠ 0, so rank(O₂)=2.
        Thus the discrete-time state is observable from two consecutive position measurements.
    """


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def get_Ak(X_nom):
    r = np.asarray(X_nom[:3], dtype=float)

    rn = np.linalg.norm(r)
    I3 = np.eye(3)
    rrT = np.outer(r, r)
    G = mu * (3.0*rrT/(rn**5) - I3/(rn**3))

    Z = np.zeros((3, 3))
    I = np.eye(3)

    Ak = np.block([[Z, I],
                   [G, Z]])

    return Ak


# REQUIRED --- Problem 3c
def get_Ck(X_nom, R_obs):
    r = np.asarray(X_nom[:3], dtype=float)
    dr = r - np.asarray(R_obs, dtype=float)
    rho = np.linalg.norm(dr)
    if rho <= 0.0:
        # Degenerate; return zeros to avoid NaNs
        grad_r = np.zeros(3)
    else:
        grad_r = dr / rho  # 1x3
    Ck = np.hstack([grad_r, np.zeros(3)]).reshape(1, 6)

    # X_nom : nominal state at time k
    # R_obs : observer position at time k
    return Ck


# REQUIRED --- Problem 3d
def plot_numerical_integration_dX():
    t0 = 0.0
    tf = 90.0 * 60.0  # 90 minutes
    t_vec = np.linspace(t0, tf, 5401)  # 1 s cadence

    dX0 = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=float)

    X_nom = integrate_trajectory(X_N_0, t_vec)
    X_pert = integrate_trajectory(X_N_0 + dX0, t_vec)

    dr = X_pert[:, :3] - X_nom[:, :3]  # (N,3)

    fig, ax = plt.subplots()
    ax.plot(t_vec/60.0, dr[:, 0], label="δr_x")
    ax.plot(t_vec/60.0, dr[:, 1], label="δr_y")
    ax.plot(t_vec/60.0, dr[:, 2], label="δr_z")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Position deviation δr [km]")
    ax.set_title("Numerical: δr(t) for small initial perturbation")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- Problem 3e
def plot_analytic_integration_dX():
    t0 = 0.0
    tf = 90.0 * 60.0
    t_vec = np.linspace(t0, tf, 5401)
    dX0 = np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=float)

    X_nom, Phi_t = integrate_nominal_and_stm(X_N_0, t_vec)

    dX_t = (Phi_t @ dX0.reshape(6, 1)).squeeze(-1)  # (N,6)
    dr = dX_t[:, :3]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t_vec/60.0, dr[:, 0], label="δr_x (STM)")
    ax.plot(t_vec/60.0, dr[:, 1], label="δr_y (STM)")
    ax.plot(t_vec/60.0, dr[:, 2], label="δr_z (STM)")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Position deviation δr [km]")
    ax.set_title("STM-based δr(t) for small initial perturbation")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- Problem 3f
def plot_critical_dX_neighborhood():
    t0 = 0.0
    tf = 90.0 * 60.0
    t_vec = np.linspace(t0, tf, 5401)
    dX0 = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=float)

    # Nonlinear truth difference
    X_nom = integrate_trajectory(X_N_0, t_vec)
    X_pert = integrate_trajectory(X_N_0 + dX0, t_vec)
    dr_truth = X_pert[:, :3] - X_nom[:, :3]
    nr_truth = np.linalg.norm(dr_truth, axis=1)

    # Linearized via STM
    X_nom2, Phi_t = integrate_nominal_and_stm(X_N_0, t_vec)
    dX_t_lin = (Phi_t @ dX0.reshape(6, 1)).squeeze(-1)
    dr_lin = dX_t_lin[:, :3]
    nr_lin = np.linalg.norm(dr_lin, axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t_vec/60.0, nr_truth, label="‖δr(t)‖ Nonlinear")
    ax.plot(t_vec/60.0, nr_lin,   label="‖δr(t)‖ STM (linearized)", linestyle="--")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("‖δr(t)‖ [km]")
    ax.set_title("Neighborhood of linearization: large initial perturbation")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- Problem 3g
def describe_neighborhood():
    return """For small perturbations, the STM linearization matches the nonlinear δr(t) closely: the
direction and amplitude agree because the first-order model captures the local flow and gravity-gradient
coupling along the nominal path. When the initial deviation is increased, higher-order terms (curvature of
the vector field and coupling through 1/‖r‖^3) become significant over an orbital timescale. The STM keeps
projecting the initial deviation through Φ(t,t0) built on the nominal, so its prediction grows biased: phase
drifts and amplitude errors accumulate, especially near perigee where the field is most nonlinear.
The plot shows divergence in ‖δr(t)‖: nonlinear truth departs from the linear prediction, evidencing the
finite neighborhood of validity for the first-order model."""


# REQUIRED --- Problem 3h
def estimtae_dX0():
    data = load_numpy_data(cur_dir + "/data/HW3-kepler-data.npy")
    y_meas = data["rho"]
    t_meas = data["t"]

    # Nominal + STM along measurement times
    X_nom, Phi_t = integrate_nominal_and_stm(X_N_0, t_meas)

    # Build stacked H and residual vector
    H_rows = []
    y_res = []
    for i, t in enumerate(t_meas):
        r_nom = X_nom[i, :3]
        R_obs = observer_position_ECI(t)
        # nominal range
        rho_nom = np.linalg.norm(r_nom - R_obs)
        # measurement residual
        y_res.append(y_meas[i] - rho_nom)
        # C_k at nominal
        Ck = get_Ck(X_nom[i, :], R_obs)  # (1x6)
        # Row: C_k * Φ_k
        H_rows.append(Ck @ Phi_t[i])     # (1x6)

    H = np.vstack(H_rows)          # (M x 6)
    y_res = np.asarray(y_res)      # (M,)

    # Solve least squares for δX0
    dX0_est, *_ = np.linalg.lstsq(H, y_res, rcond=None)
    return dX0_est.astype(float)
    return dX0_estimated


# REQUIRED --- Problem 3i
def explain_approach():
    return """We linearize the range model about the nominal: y_k = ρ(r_k) with r_k ≈ r_nom,k + δr_k,
δX_k = Φ_k δX0. The measurement Jacobian is C_k = [ (r_nom,k − r_obs,k)^T / ρ_nom,k , 0_{1×3} ].
Thus y_k − ρ_nom,k ≈ (C_k Φ_k) δX0. Stacking all samples yields a linear least-squares problem:
min_δX0 || y_res − H δX0 ||₂, where rows of H are C_k Φ_k. We integrate once along the nominal
and its STM to get X_nom(t_k), Φ_k, build H and the residual vector y_res, then solve with
np.linalg.lstsq. This provides the best linearized estimate of the initial deviation δX0 given
range-only data and the chosen nominal trajectory."""


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    # 1a
    # with open("./outputs/text/s01a.txt", "w", encoding="utf-8") as f:
    #     f.write(propogate_CT_LTI_numerically)
    # 1b
    # with open("./outputs/text/s01b.txt", "w", encoding="utf-8") as f:
    #     f.write(propogate_CT_LTI_analytically)
    # 1c
    # with open("./outputs/text/s01c.txt", "w", encoding="utf-8") as f:
    #     f.write(propogate_DT_LTI_analytically)
    # 1d
    # with open("./outputs/text/s01d.txt", "w", encoding="utf-8") as f:
    #     f.write(propagate_LTV_system_numerically)
    # 1e
    with open("./outputs/text/s01e.txt", "w", encoding="utf-8") as f:
        f.write(str(max_allowable_sampling_time(A)))

    # Problem 2
    # 2b
    plot_trajectories().savefig("./outputs/figures/s02b.png", dpi=300)
    # 2c
    with open("./outputs/text/s02c.txt", "w", encoding="utf-8") as f:
        f.write(str(describe_propagation_methods()))
    # 2d
    with open("./outputs/text/s02d.txt", "w", encoding="utf-8") as f:
        f.write(str(determine_x0()))
    # 2e
    with open("./outputs/text/s02e.txt", "w", encoding="utf-8") as f:
        f.write(str(describe_observability()))

    # Problem 3
    # 3b
    with open("./outputs/text/s03b.txt", "w", encoding="utf-8") as f:
        f.write(str(get_Ak(X_N_0)))
    # 3c
    with open("./outputs/text/s03c.txt", "w", encoding="utf-8") as f:
        f.write(str(get_Ck(X_N_0, r_obs_ecef0)))
    # 3d
    plot_numerical_integration_dX().savefig("./outputs/figures/s03d.png", dpi=300)
    # 3e
    plot_analytic_integration_dX().savefig("./outputs/figures/s03e.png", dpi=300)
    # 3f
    plot_critical_dX_neighborhood().savefig("./outputs/figures/s03f.png", dpi=300)
    # 3g
    with open("./outputs/text/s03g.txt", "w", encoding="utf-8") as f:
        f.write(str(describe_neighborhood()))
    # 3h
    with open("./outputs/text/s03h.txt", "w", encoding="utf-8") as f:
        f.write(str(estimtae_dX0()))
    # 3i
    with open("./outputs/text/s03i.txt", "w", encoding="utf-8") as f:
        f.write(str(explain_approach()))


if __name__ == "__main__":
    main()
