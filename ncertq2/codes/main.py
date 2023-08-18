import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import binom
simlen=40
n=2
p=1/2

# USING PROBABILITY MASS FUNCTION FOR BINOMIAL DISTRIBUTION
binom_dist=binom(n,p)
# probability of getting two heads in two coin flips
two_head_prob = binom_dist.pmf(2)
print(1-two_head_prob)
print()
# USING BINOMIAL DISTRIBUTION
data_binom = binom.rvs(n,p,size=simlen) #Simulating the event of two heads
print(data_binom)
err_ind = np.nonzero(data_binom <=1) #checking probability condition of atmost one head
err_n = np.size(err_ind) #computing the probability
print(err_n/simlen)

print()
# USING BERNOULLI DISTRIBUTION
data_bern_mat = bernoulli.rvs(p,size=(n,simlen))
data_binom=np.sum(data_bern_mat, axis=0)
print(data_binom)
err_ind = np.nonzero(data_binom <2) #checking probability condition
err_n = np.size(err_ind) #computing the probability
print(err_n/simlen)

