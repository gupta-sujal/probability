import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import binom
n=2
simlen=int(1e2)

#Probability of the event
prob=1/2
#Generating sample date using Bernoulli r.v.
data_bern_mat = bernoulli.rvs(prob,size=(n,simlen))
data_binom=np.sum(data_bern_mat, axis=0)
print(data_binom)
err_ind = np.nonzero(data_binom <=1) #checking probability condition of atmost one head
err_n = np.size(err_ind) #computing the probability
print("the practical probability of atmost one head is",err_n/simlen)
print()

# using binomial to calculate theoretical probability of atmost one head.
# USING PROBABILITY MASS FUNCTION FOR BINOMIAL DISTRIBUTION
binom_dist=binom(n,prob)
# probability of getting two heads in two coin flips
two_head_prob = binom_dist.pmf(2)
prob_atmost_one_head=1-two_head_prob
print(prob_atmost_one_head)


