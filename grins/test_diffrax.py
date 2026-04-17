import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=15"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"

import jax as jx
import jax.numpy as jnp
import numpy as np

# import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, SteadyStateEvent  # noqa: E402
from jax import vmap, pmap  # noqa: E402
import pandas as pd
import time

from multiprocessing import Pool

# # Defining a custom addition function
# def add(a, b):
#     return a + b


# DEfining the ODE system
def vector_field(t, y, args):
    prey, predator = y
    α, β, γ, δ = args
    d_prey = α * prey - β * prey * predator
    d_predator = -γ * predator + δ * prey * predator
    # d_predator = add(-γ * predator, δ * prey * predator)
    d_y = d_prey, d_predator
    return d_y


# Defining the shifted hills functions
# Positive Shifted Hill function
def psH(nod, fld, thr, hill):
    return (fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))) / fld


# Negative Shifted Hill function
def nsH(nod, fld, thr, hill):
    return fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))


# Function for Toggle Switch
def toggle_switch(t, y, args):
    A, B = y
    # (
    #     Prod_A,
    #     Prod_B,
    #     Deg_A,
    #     Deg_B,
    #     InhFld_B_A,
    #     Thr_B_A,
    #     Hill_B_A,
    #     InhFld_A_B,
    #     Thr_A_B,
    #     Hill_A_B,
    # ) = args
    # For RACIPE paramn
    (
        Prod_A,
        Prod_B,
        Deg_A,
        Deg_B,
        Thr_B_A,
        Hill_B_A,
        InhFld_B_A,
        Thr_A_B,
        Hill_A_B,
        InhFld_A_B,
    ) = args
    d_A = Prod_A * nsH(B, InhFld_B_A, Thr_B_A, Hill_B_A) - Deg_A * A
    d_B = Prod_B * nsH(A, InhFld_A_B, Thr_A_B, Hill_A_B) - Deg_B * B
    d_y = d_A, d_B
    return d_y


# Wrapper function to sovle the ODE
def solve_ode(vector_field, solver, t0, t1, dt0, y0, args, saveat, controller):
    # term = ODETerm(vector_field)
    # solver = Tsit5()
    # t0 = 0
    # t1 = 140
    # dt0 = 0.1
    y0 = tuple(y0)
    # args = (0.1, 0.02, 0.4, 0.02)
    # saveat = SaveAt(t1=True)
    sol = diffeqsolve(
        ODETerm(vector_field),
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=args,
        saveat=saveat,
        max_steps=1000,
        stepsize_controller=controller,
        discrete_terminating_event=SteadyStateEvent(),
        # throw=None,
    )
    sol = [s[-1] for s in sol.ys]
    return sol


# term = ODETerm(vector_field)
# solver = Tsit5()
# t0 = 0
# t1 = 140
# dt0 = 0.1
# max_steps = None
# controller = PIDController(rtol=1e-4, atol=1e-6)
# args = (0.1, 0.02, 0.4, 0.02)
# saveat = SaveAt(t1=True)

# # Make a 2xnum_inital of all the random initial conditions
# y0 = jnp.array(np.random.rand(10000, 2) * 10)
# # # Get shape of the initial conditions
# print(y0.shape)

# # Track the time taken to solve the ODEs
# print("Solving the ODEs")
# start = time.time()
# # Map the solve_ode function over the list of initial conditions
# # sol_li = vmap(lambda i: solve_ode(vector_field, t0, t1, dt0, i, args, saveat))(y0)
# sol_li = vmap(
#     lambda i: solve_ode(vector_field, solver, t0, t1, dt0, i, args, saveat, controller),
# )(y0)
# # Print the time taken to solve the ODEs
# print(f"Time taken to solve the ODEs: {time.time() - start}")

# # Convert the solution list to a numpy array
# sol_li = jnp.array(sol_li).T
# # Convert the solution list to a data frame
# sol_df = pd.DataFrame(sol_li, columns=["Prey", "Predator"])
# print(sol_li.shape)
# print(sol_df)
# print(sol_df.describe())


# Defining the problem for the toggle switch
ts_term = ODETerm(toggle_switch)
solver = Tsit5()
t0 = 0
t1 = 140
dt0 = 0.1
max_steps = None
controller = PIDController(rtol=1e-5, atol=1e-6)
# RACIPE parameters: 8.980273,55.355989,0.935186,0.767823,34.754825,5.000000,0.016662,14.209517,4.000000,0.023031
# args = (
#     26.11365514714271,
#     71.88358915410936,
#     0.5858103133738041,
#     0.576099329534918,
#     0.04951568856870413,
#     62.75370424835297,
#     4.0,
#     0.010144997482053913,
#     2.8172455595347636,
#     6.0,
# )
args = (
    8.980273,
    55.355989,
    0.935186,
    0.767823,
    34.754825,
    5.000000,
    0.016662,
    14.209517,
    4.000000,
    0.023031,
)
saveat = SaveAt(t1=True)
# Add steady state temrination event
ss_event = SteadyStateEvent()

# y0 = (40.389327354729176, 60.03178594075143)
# Repeat y0 10 times and get a 2x10 array
# y0 = jnp.array([y0] * 10000)
# Rnaomly generate 10000 initial conditions
y0 = jnp.array(np.random.rand(10000000, 2) * 100)
print(y0.shape)


# Track the time taken to solve the ODEs
print("Solving the ODEs")
start = time.time()
# Map the solve_ode function over the list of initial conditions
# sol_li = vmap(lambda i: solve_ode(vector_field, t0, t1, dt0, i, args, saveat))(y0)
sol_li = vmap(
    lambda i: solve_ode(
        toggle_switch, solver, t0, t1, dt0, i, args, saveat, controller
    ),
)(y0)
# Print the time taken to solve the ODEs
print(f"Time taken to solve the ODEs: {time.time() - start}")

# Convert the solution list to a numpy array
sol_li = jnp.array(sol_li).T
# Convert the solution list to a data frame
sol_df = pd.DataFrame(sol_li, columns=["A", "B"])
print(sol_li.shape)
print(sol_df)
print(sol_df.describe())
