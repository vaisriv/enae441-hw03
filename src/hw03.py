import os

import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
# Gravitational parameter for Earth in km^3/s^2
mu = 398600  # km^3/s^2
omega_EN = 7.2921150e-5  # rad /s

# Problem 2
X_0_spring = np.array([1, 0])  # m, m/s

X_N_0 = np.array([7000, 0, 0, 0, 7.5, 3.5])  # km, km/s
dX_N_0 = np.array([30, 0, 0, 0, 0, 0.1])  # km, km/s

cur_dir = os.path.dirname(__file__)
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
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1b
def propogate_CT_LTI_analytically(X_0, A, t_vec):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1c
def propogate_DT_LTI_analytically(X_0, A, dt, k_max):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    return X_t


# REQUIRED --- 1d
def propagate_LTV_system_numerically(X_0, x_dot_fcn, A_fcn, t_vec):
    # Return trajectory and STM over time where
    # np.shape(X_t_vec) = (len(t_vec), len(X_0))
    # np.shape(phi_t_vec) = (len(t_vec), len(X_0), len(X_0))

    return X_t_vec, phi_t_vec


# REQUIRED --- 1e
def max_allowable_sampling_time(A):
    # compute the maximum allowable sampling time
    return dt_max


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2b
def plot_trajectories():
    return fig


# REQUIRED --- Problem 2c
def describe_propagation_methods():
    return """
        Write your answer here.
    """


# REQUIRED --- Problem 2d
def determine_x0():
    data = load_numpy_data(cur_dir + "/HW3-spring-data.npy")
    t_meas = data["t"]
    y_meas = data["y"]

    return X0  # dimension (2,)


# REQUIRED --- Problem 2e
def describe_observability():
    # compute how many measurements you need and return value and discription
    num_measurements = None

    return f"""
        {num_measurements} are needed to observe the state, because ...
    """


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def get_Ak(X_nom):
    return Ak


# REQUIRED --- Problem 3c
def get_Ck(X_nom, R_obs):
    # X_nom : nominal state at time k
    # R_obs : observer position at time k
    return Ck


# REQUIRED --- Problem 3d
def plot_numerical_integration_dX():
    return fig


# REQUIRED --- Problem 3e
def plot_analytic_integration_dX():
    return fig


# REQUIRED --- Problem 3f
def plot_critical_dX_neighborhood():
    return fig


# REQUIRED --- Problem 3g
def describe_propagation_methods():
    return """Write your answer here"""


# REQUIRED --- Problem 3h
def estimtae_dX0():
    data = load_numpy_data(cur_dir + "/HW3-kepler-data.npy")
    y_meas = data["rho"]
    t_meas = data["t"]

    return dX0_estimated


# REQUIRED --- Problem 3hi
def explain_approach():
    return """Write your answer here"""


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    propogate_CT_LTI_numerically
    propogate_CT_LTI_analytically
    propogate_DT_LTI_analytically
    propagate_LTV_system_numerically
    max_allowable_sampling_time

    # Problem 2
    plot_trajectories()
    describe_propagation_methods()
    determine_x0()
    describe_observability()

    # Problem 3
    get_Ak(X_N_0)
    get_Ck(X_N_0)
    plot_numerical_integration_dX()
    plot_analytic_integration_dX()
    plot_critical_dX_neighborhood()
    describe_propagation_methods()
    estimtae_dX0()
    explain_approach()
    plt.show()


if __name__ == "__main__":
    main()
