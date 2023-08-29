import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

n = 2  # Number of coin tosses
p = 0.5  # Probability of getting a head

x = np.arange(0, n+1)
pmf_values = binom.pmf(x, n, p)
plt.stem(x, pmf_values)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.title('PMF of Number of Heads in Two Coin Tosses')
plt.xticks(x)
plt.grid(True)
plt.show()

cdf_values = binom.cdf(x, n, p)
plt.figure(1)
plt.plot(x, cdf_values, marker='o', linestyle='-', drawstyle='steps-post')
plt.xlabel('Number of Heads')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Number of Heads in Two Coin Tosses')
plt.xticks(x)
plt.grid(True)
plt.savefig('/home/sujalgupat484/Desktop/probability/ncertq2/figs/figure2.png')

# Generating vectors of successes
x1 = np.random.randint(0, 2, size=10000)
x2 = np.random.randint(0, 2, size=10000)

# Calculate successful outcomes
successful_outcomes = x1 + x2 

# Count occurrences of each outcome
counts = np.bincount(successful_outcomes, minlength=3)

# Define possible values of h
possible_values_of_h = np.arange(0, 3)

# Calculate probabilities
probabilities = counts / 10000

#for plotting and simulation
# Sample size
simlen = 10000

# Possible outcomes
k_values = np.arange(0, 3)

# Generate X1 and X2 without explicit loops
y = np.random.randint(0, 2, size=(2, simlen))

# Calculate X without loops
X = np.sum(y, axis=0)

# Find the frequency of each outcome
unique, counts = np.unique(X, return_counts=True)

# Simulated probability
psim = counts / simlen

X_axis = x = np.arange(0, n+1)


# Plotting
plt.figure(2)
plt.stem(X_axis, psim, label='Simulation',linefmt='blue')
plt.stem(X_axis, pmf_values, label='Analysis',linefmt='orange')
plt.xlabel('$A$')  # Use 'k' instead of 'n'
plt.ylabel('$p_{A}(k)$')  # Use 'k' instead of 'n'
plt.legend()
plt.title('Comaprision of theoretical and calculated PMF values')
plt.grid()
plt.savefig('/home/sujalgupat484/Desktop/probability/ncertq2/figs/figure1.png')
