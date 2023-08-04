import numpy as np
from sympy import init_printing
init_printing(use_unicode=True)
from sympy import symbols
from sympy.matrices import Matrix
from numpy import linalg as LA
A=np.array([1,-1])
B=np.array([-4,6])
C=np.array([-3,-5])
D=C-B
a=np.linalg.norm(C-B)
b=np.linalg.norm(A-C)
c=np.linalg.norm(A-B)
I=np.array([(a*A[0]+b*B[0]+c*C[0])/(a+b+c),(a*A[1]+b*B[1]+c*C[1])/(a+b+c)])
print("the incentre coordiantes are",I)
# by using the data from previous question we have inradius value r
radius=1.8968927705299559
p=pow(np.linalg.norm(C-B),2)
q=2*(D@(I-B))
r=pow(np.linalg.norm(I-B),2)-radius*radius

Discre=q*q-4*p*r

print("the Value of discriminant is ",abs(round(Discre,6)))
#  so the value of Discrimant is extremely small and tends to zero
#  the discriminant value rounded off to 6 decimal places is also zero
#  so it proves that there is only one solution of the point

#  the value of point is x=B+k(C-B)
k=((I-B)@(C-B))/((C-B)@(C-B))
print("the value of parameter k is ",k)
D3=B+k*(C-B)
print("the point of tangency of incircle by side BC is ",D3)
#  to check we also check the value of dot product of ID3 and BC
print("the dot product of ID3 and BC",abs(round(((D3-I)@(C-B),6))))
#  so this comes out to be zero
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")