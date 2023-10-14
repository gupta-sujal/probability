import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Define the shape and scale parameters
shape_parameter = 2.0  # Shape (k) parameter
scale_parameter = 1.0  # Scale (theta) parameter

# Generate random samples from a gamma distribution
samples = np.random.gamma(shape_parameter, scale_parameter, size=10000)
alpha=1
lam=1
mean,variance=gamma.stats(alpha, loc=0, scale=1/lam, moments='mv')
print(mean,variance)

alpha=-1
lam=1
mean,variance=gamma.stats(alpha, loc=0, scale=1/lam, moments='mv')
print(mean,variance)
print("not a number(nan) values come as output for alpha<0")

