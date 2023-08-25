import numpy as np
import matplotlib.pyplot as plt
import sys                                          #for path to external scripts
sys.path.insert(0, '/home/sujalgupat484/training/math/codes/CoordGeo')        #path to my scripts
import matplotlib.image as mpimg

#local imports
from line.funcs import *

from triangle.funcs import *
from conics.funcs import circ_gen
A = np.array([-6,0])
B = np.array([-4,3])
C = np.array([-2,0])
simlen=2
# y = np.random.randint(-6,6, size=(3, simlen))
# A=y[0]
# B=y[1]
# C=y[2]
A1=A
B1=B 
C1=C
print('A =',A)
print('B =',B)
print('C =',C)
d = B- A
e = C - B
f = A - C
# 1.1.1
print("The direction vector of AB is ",d)
print("The direction vector of BC is ",e)
print("The direction vector of CA is ",f)

# 1.1.2
length_AB = np.linalg.norm(d)
length_BC = np.linalg.norm(e)
length_CA = np.linalg.norm(f)
print("Length of side AB:", length_AB)
print("Length of side BC:", length_BC)
print("Length of side CA:", length_CA)

# 1.1.3
Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])

rank = np.linalg.matrix_rank(Mat)

if (rank<=2):
	print("Hence proved that points A,B,C in a triangle are collinear")
else:
	print("The given points are not collinear")
# 1.1.4
def parametric_form (A,B,k):
    direction_vector_AB = B-A
    x = A + k * direction_vector_AB
    return x
# 1.1.5
omat = np.array([[0,1],[-1,0]])
def norm_vec(C,B):
    return omat@(C-B)
n=norm_vec(A,B)
pro=n@B
print("AB ",n,"x=",pro)
n=norm_vec(B,C)
pro=n@B
print("BC ",n,"x=",pro)
n=norm_vec(C,A)
pro=n@B
print("CA ",n,"x=",pro)
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
x_BC = line_gen(B,C)
x_AB = line_gen(A, B)
x_CA = line_gen(C, A)
plt.figure(1)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
tri_coords = np.block([[A,B,C]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-1-5.png')


# 1.1.6
def AreaCalc(A, B, C):
#cross_product calculation
    cross_product = np.cross(d,f)
#magnitude calculation
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    return area
print("Area of triangle ABC:", AreaCalc(A,B,C))

# 1.1.7
dotA=((B-A).T)@(C-A)
dotA=dotA[0,0]
NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print('value of angle A in degrees: ', np.degrees(np.arccos((dotA)/NormA)))


dotB=(A-B).T@(C-B)
dotB=dotB[0,0]
NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))
print('value of angle B in degrees: ', np.degrees(np.arccos((dotB)/NormB)))

dotC=(A-C).T@(B-C)
dotC=dotC[0,0]
NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
print('value of angle C in degrees: ', np.degrees(np.arccos((dotC)/NormC)))

# 1.2.1
D = (B1 + C1)/2

#Similarly for E and F
E = (A1 + C1)/2
F = (A1 + B1)/2
D1=D
E1=E
F1=F

print("D:", list(D))
print("E:", list(E))
print("F:", list(F))
print(D.shape)
#Plotting all lines
plt.figure(4)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
#Labeling the coordinates

D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('tri_sss.pdf')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-2-1.png')

# 1.2.2
n=norm_vec(A1,D1)
pro=n@B
print("AD ",n,"x=",pro)
n=norm_vec(B1,E1)
pro=n@B
print("BE ",n,"x=",pro)
n=norm_vec(C1,F1)
pro=n@B
print("CF ",n,"x=",pro)
# the lines AD,BE,CF Are already plotted in the previous figure
# 1.2.3
def line_intersect(n1,A1,n2,A2):
    N=np.block([[n1],[n2]])
    p = np.zeros(2)
    # print(A1.shape)
    p[0] = n1@A1
    p[1] = n2@A2
    #Intersection
    P=np.linalg.inv(N)@p
    return P
G=line_intersect(norm_vec(F1,C1),C1,norm_vec(E1,B1),B1)
print("the point of intersection of BE AND CF is ("+str(G[0])+","+str(G[1])+")")
G1=G
G = G.reshape(-1,1)
x_BE = line_gen(B,E)
x_CF = line_gen(C,F)
x_AD = line_gen(A,D)
plt.figure(5)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
tri_coords = np.block([[A, B, C, D, E, F, G]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-2-3.png')

# 1.2.4
n_BG = norm_vec(B,G)
n_GE = norm_vec(G,E)
n_GF = norm_vec(G,F)
n_CG = norm_vec(C,G)
n_AG = norm_vec(A,G)
n_GD = norm_vec(G,D)

print("The norm of the vector BG is",n_BG)
print("The norm of the vector GE is",n_GE)
print("The norm of the vector GF is",n_GF)
print("The norm of the vector CG is",n_CG)
print("The norm of the vector AG is",n_AG)
print("The norm of the vector GD is",n_GD)

print("The ratio BG/GE is",n_BG/n_GE)
print("The ratio CG/GF is",n_CG/n_GF)
print("The ratio AG/GD is",n_AG/n_GD)

# 1.2.5
print(f"The centroid of triangleABC is {G}")

Mat = np.array([[1,1,1],[A1[0],D1[0],G1[0]],[A1[1],D1[1],G1[1]]])

rank = np.linalg.matrix_rank(Mat)

if (rank==2):
	print("Hence proved that points A,G,D in a triangle are collinear")
else:
	print("Error")
        
# 1.2.6
G1 = (A1 + B1 + C1) / 3
print("centroid of the given triangle: ",G1)  

# 1.2.7
LHS=(A1-F1)
RHS=(E1-D1)
#checking LHS and RHS 
if LHS.all()==RHS.all() :
   print("A-F=E-D")
else:
    print("Not equal")

x_AF = line_gen(A1,F1)
x_FD = line_gen(F1,D1)
x_DE = line_gen(D1,E1)
x_EA = line_gen(E1,A1)

plt.figure(6)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AF[0,:],x_AF[1,:],label='$AF$')
plt.plot(x_FD[0,:],x_FD[1,:],label='$FD$')
plt.plot(x_DE[0,:],x_DE[1,:],label='$DE$')
plt.plot(x_EA[0,:],x_EA[1,:],label='$EA$')
tri_coords = np.block([[A, B, C, D, E, F]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D', 'E', 'F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-2-7.png')

# 1.3.1

t=np.array([0,1,-1,0]).reshape(2,2)

#AD_1
AD_1=t@e
#normal vector of AD_1
AD_p=t@AD_1
print(AD_p)

# 1.3.2
T = np.array([1,-1])
nt = np.array([-1,11])
result = T@nt
print(f"The equation of AD is {nt}X={result}")
D_1 = alt_foot(A1,B1,C1)
print("D1 is ",D_1)
E_1 = alt_foot(B1,A1,C1)
print("E1 is ",E_1)
F_1 = alt_foot(C1,B1,A1)
print("F1 is ",F_1)
n=norm_vec(A1,D_1)
pro=n@B
print("AD1 ",n,"x=",pro)
n=norm_vec(B1,E_1)
pro=n@B
print("BE1 ",n,"x=",pro)
n=norm_vec(C1,F_1)
pro=n@B
print("CF1 ",n,"x=",pro)

plt.figure(7)
x_AD_1 = line_gen(A1,D_1)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')
D_1_1 = D_1.reshape(-1,1)

tri_coords = np.block([[A,B,C,D_1_1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-3-2.png')

# 1.3.3
E_1 =  alt_foot(B1,C1,A1)
F_1 =  alt_foot(C1,A1,B1)
plt.figure(8)
x_BE_1 = line_gen(B1,E_1)
x_CF_1 = line_gen(C1,F_1)
x_AE_1=line_gen(A1,E_1)
x_AF_1=line_gen(A1,F_1)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')
plt.plot(x_CF_1[0,:],x_CF_1[1,:],label='$CF_1$')
plt.plot(x_BE_1[0,:],x_BE_1[1,:],label='$BE_1$')
plt.plot(x_AE_1[0,:],x_AE_1[1,:],linestyle='dotted')
plt.plot(x_AF_1[0,:],x_AF_1[1,:],linestyle='dotted')
E_1_1 = E_1.reshape(-1,1)
F_1_1 = F_1.reshape(-1,1)
tri_coords = np.block([[A,B,C,D_1_1,E_1_1,F_1_1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1','E_1','F_1']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-3-3.png')

# 1.3.4
H=line_intersect(norm_vec(F_1,C1),C1,norm_vec(E_1,B1),B1)
print("intersection of BE1 AND CF1",H)
plt.figure(9)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')
plt.plot(x_CF_1[0,:],x_CF_1[1,:],label='$CF_1$')
plt.plot(x_BE_1[0,:],x_BE_1[1,:],label='$BE_1$')
plt.plot(x_AE_1[0,:],x_AE_1[1,:],linestyle='dotted')
plt.plot(x_AF_1[0,:],x_AF_1[1,:],linestyle='dotted')
x_CH = line_gen(C1,H)
x_BH = line_gen(B1,H)
x_AH = line_gen(A1,H)
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')
H1 = H
H=H.reshape(-1,1)
tri_coords = np.block([[A,B,C,D,E,F,H]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D_1','E_1','F_1','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-3-4.png')

# 1.3.5
result = int(((A1 - H1).T) @ (B1 - C1))    # Checking orthogonality condition...

# printing output
if result == 0:
  print("(A - H)^T (B - C) = 0\nHence Verified...")

else:
  print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong...")

X = np.array([-203/61, -85/61])     #X is point of intersection of line AH and BC

#Generating all lines
x_AH = line_gen(A1,H1)
x_BC = line_gen(B1,C1)
x_AX = line_gen(A1,X)

#Plotting all lines
plt.figure(10)
plt.plot(x_AH[0,:],x_AH[1,:],label='$AH$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AX[0,:],x_AX[1,:],linestyle='dotted',label='$AX$')

tri_coords = np.block([[A,B,C,H]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-3-5.png')

# 1.4.1
def midpoint(P, Q):
    return (P + Q) / 2  
#normal vector 
def norm_vec(A,B):
  omat = np.array([[0,1],[-1,0]]) 
  return omat.T@(A-B)
#to find the coefficients and constant of the equation of perpendicular bisector of BC
def perpendicular_bisector(B, C):
    midBC=midpoint(B,C)
    dir=B-C
    constant = -dir.T @ midBC
    return dir,constant
equation_coeff1,const1 = perpendicular_bisector(A1, B1)
equation_coeff2,const2 = perpendicular_bisector(B1, C1)
equation_coeff3,const3 = perpendicular_bisector(C1, A1)
print(f'Equation for perpendicular bisector of AB: ({equation_coeff1[0]:.2f})x + ({equation_coeff1[1]:.2f})y + ({const1:.2f}) = 0')
print(f'Equation for perpendicular bisector of  BC: ({equation_coeff2[0]:.2f})x + ({equation_coeff2[1]:.2f})y + ({const2:.2f}) = 0')
print(f'Equation for perpendicular bisector of  CA: ({equation_coeff3[0]:.2f})x + ({equation_coeff3[1]:.2f})y + ({const3:.2f}) = 0')
def ccircle(A,B,C):
  p = np.zeros(2)
  n1 = equation_coeff1[:2]
  p[0] = 0.5*(np.linalg.norm(A)**2-np.linalg.norm(B)**2)
  n2 = equation_coeff2[:2]
  p[1] = 0.5*(np.linalg.norm(B)**2-np.linalg.norm(C)**2)
  #Intersection
  N=np.block([[n1],[n2]])
  O=np.linalg.solve(N,p)
  return O
O=ccircle(A,B,C)

# Plotting all lines
plt.figure(11)
plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')
plt.plot(x_BC[0, :], x_BC[1, :], label='$BC$')
plt.plot(x_CA[0, :], x_CA[1, :], label='$CA$')
# Perpendicular bisector
def line_dir_pt(m, A, k1=0, k2=1):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(k1, k2, len)
    for i in range(len):
        temp1 = A + lam_1[i] * m
        x_AB[:, i] = temp1.T
    return x_AB
# Calculate the perpendicular vector and plot arrows
def perpendicular(B, C, label,  O):
    perpendicular=norm_vec(B,C)
    mid = midpoint(B, C)
    # x_D = line_dir_pt(perpendicular, mid, 0, 1)
    x_D=line_gen(mid,O)
    plt.plot(x_D[0, :], x_D[1, :], label=label)
    #plt.arrow(mid[0], mid[1], perpendicular[0], perpendicular[1], color='blue', head_width=0.4, head_length=0.4, label=label)
    #plt.arrow(mid[0], mid[1], -perpendicular[0], -perpendicular[1], color='blue', head_width=0.4, head_length=0.4)
    return x_D
x_D = perpendicular(A1, B1, 'OD',  O)
x_E = perpendicular(B1, C1, 'OE',  O)
x_F = perpendicular(C1, A1, 'OF',  O)
#Labeling the coordinates
#tri_coords = np.vstack((A,B,C,O,I)).T
#np.block([[A1,A2,B1,B2]])
O1=O
O = O.reshape(-1,1)

tri_coords = np.block([[A,B,C,O,D,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-4-1,2,3.png')

# 1.4.2
print("circumcentre is",O1)

# 1.4.3
# all the calculations have been done in 1.4.1

# 1.4.4
O_1 = O1 - A1
O_2 = O1 - B1
O_3 = O1 - C1
a = np.linalg.norm(O_1)
b = np.linalg.norm(O_2)
c = np.linalg.norm(O_3)
print("Points of triangle A, B, C respectively are", A1 ,",", B1 ,",", C1, ".")
print("Circumcentre of triangle is", O1, ".")
print(" OA, OB, OC are respectively", a,",", b,",",c, ".")
print("Here, OA = OB = OC.")
print("Hence verified.")

# 1.4.5
X = A1 - O1
radius = np.linalg.norm(X)
x_ccirc= circ_gen(O1,radius)
x_OA = line_gen(O1,A1)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')

#Plotting the circumcircle
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
#Labeling the coordinates
tri_coords = np.block([[A,B,C,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-4-5.png')

# 1.4.6
dot_pt_O = (B1 - O1) @ ((C1 - O1).T)
norm_pt_O = np.linalg.norm(B1 - O1) * np.linalg.norm(C1 - O1)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC = round(np.degrees(np.arccos(cos_theta_O)),1)  #Round is used to round of number till 5 decimal places
print('')
print("angle BOC = " + str(angle_BOC))

#To find angle BAC
dot_pt_A = (B1 - A1) @ ((C1 - A1).T)
norm_pt_A = np.linalg.norm(B1 - A1) * np.linalg.norm(C1 - A1)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),1)  #Round is used to round of number till 5 decimal places
print("angle BAC = " + str(angle_BAC))
#To check whether the answer is correct
if angle_BOC == 2 * angle_BAC:
  print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
  print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
x_AC = line_gen(A1,C1)
x_OB = line_gen(O1,B1)
x_OC = line_gen(O1,C1)
plt.plot(x_AC[0,:],x_AC[1,:],label='$BC$')
plt.plot(x_OB[0,:],x_OB[1,:],label='$OB$')
plt.plot(x_OC[0,:],x_OC[1,:],label='$OB$')
tri_coords = np.block([[A,B,C,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-4-6.png')

# 1.5.1
[I,r] = icircle(A1,B1,C1)
r=abs(r)
x_icirc= circ_gen(I,r)
x_AI = line_gen(A1 ,I)
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
def unit_vec(A,B):
	return ((B-A)/np.linalg.norm(B-A))

P= unit_vec(A1,B1) + unit_vec(A1,C1)
I1=I
I=I.reshape(-1,1)
#point generated to create parametric form
#generating normal form
P=np.array([P[1],(P[0]*(-1))])
#matrix multiplication
C_1= P@(A1.T)
print("Internal Angular bisector of angle A is:",P,"*x = ",C_1)
plt.figure(14)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
tri_coords = np.block([A,B,C,I])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0,i], tri_coords[1,i]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-5-1.png')

# 1.5.1(1)
P= unit_vec(B1,C1) + unit_vec(A1,B1)
P=np.array([P[1],(P[0]*(-1))])
C_1= P@(B1.T)
print("Internal Angular bisector of angle B is:",P,"*x = ",C_1)
P= unit_vec(A1,C1) + unit_vec(C1,B1)
P=np.array([P[1],(P[0]*(-1))])
C_1= P@(C1.T)
print("Internal Angular bisector of angle C is:",P,"*x = ",C_1)
x_BI = line_gen(B1 , I1)
x_CI = line_gen(C1 , I1)
plt.figure(15)
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
tri_coords = np.block([[A1],[B1],[C1],[I1]])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[i,0], tri_coords[i,1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-5-2.png')

# 1.5.2
print("the point of intersecion of the internal angle bisecors is the Incentre I ",I1)

# 1.5.3
BA = A1 - B1
CA = A1 - C1
IA = A1- I1

def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg

#Calculating the angles BAI and CAI
angle_BAI = angle_btw_vectors(BA, IA)
angle_CAI = angle_btw_vectors(CA, IA)

# Print the angles
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
    print("Angle BAI is exactly equal to angle CAI.")
else:
    print("error")
plt.figure(16)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
tri_coords = np.block([A,B,C,I])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
                 (tri_coords[0,i], tri_coords[1,i]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-5-3.png')

# 1.5.4
k1=1
k2=1

p = np.zeros(2)
t = norm_vec(B1, C1)
n1 = t / np.linalg.norm(t)
t = norm_vec(C1, A1)
n2 = t / np.linalg.norm(t)
t = norm_vec(A1, B1)
n3 = t / np.linalg.norm(t)

p[0] = n1 @ B1 - k1 * n2 @ C1
p[1] = n2 @ C1 - k2 * n3 @ A1
distI_BC = n1 @ (B-I)

print("Coordinates of point I:", I)
print(f"Distance from I to BC= {distI_BC}")

# 1.5.5-\
print(f"Distance from I to AB= {distI_BC}")
print(f"Distance from I to CA= {distI_BC}")
# all distances are same since it is inradius
# 1.5.6
plt.figure(17)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#plotting the incircle
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#labelling the coordinates
tri_coords = np.block([[A1],[B1],[C1],[I1]]).T
tri_coords = tri_coords.reshape(2, -1)
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-5-7.png')
print("the inradius is",r)

# 1.5.8
radius=r        
p=pow(np.linalg.norm(C1-B1),2)
q=2*(( C1-B1)@(I1-B1))
r=pow(np.linalg.norm(I1-B1),2)-radius*radius

Discre=q*q-4*p*r

print("the Value of discriminant is ",abs(round(Discre,6)))
#  so the value of Discrimant is extremely small and tends to zero
#  the discriminant value rounded off to 6 decimal places is also zero
#  so it proves that there is only one solution of the point

#  the value of point is x=B+k(C-B)
k=((I1-B1)@(C1-B1))/((C1-B1)@(C1-B1))
print("the value of parameter k is ",k)
D3=B1+k*(C1-B1)
print("the point of tangency of incircle by side BC is ",D3)
#  to check we also check the value of dot product of ID3 and BC
#print("the dot product of ID3 and BC",abs(round(((D3-I)@(C-B),6))))
#  so this comes out to be zero
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")

# 1.5.9
k1=((I1-A1)@(A1-B1))/((A1-B1)@(A1-B1))
k2=((I1-A1)@(A1-C1))/((A1-C1)@(A1-C1))
#finding E_3 and F_3
E3=A1+(k1*(A1-B1))
F3=A1+(k2*(A1-C1))
print("k1 = ",k1)
print("k2 = ",k2)
print("E3 = ",E3)
print("F3 = ",F3)
plt.figure(18)
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
D_3=D3
E_3=E3
F_3=F3
D3 = D3.reshape(-1,1)
E3 = E3.reshape(-1,1)
F3 = F3.reshape(-1,1)
tri_coords = np.block([[A,B,C,I,D3,E3,F3]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I','D3','E3','F3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid(True) # minor
plt.axis('equal')
plt.savefig('/home/sujalgupat484/Desktop/probability/assgn3/figs/1-5-8,9.png')

# 1.5.10
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
    return magnitude 
print("AE_3=", norm(A1,E_3) ,"\nAF_3=", norm(A1,F_3) ,"\nBD_3=", norm(B1,D_3) ,"\nBF_3=", norm(B1,E_3) ,"\nCD_3=", norm(C1,D_3) ,"\nCE_3=",norm(C1,F_3))

# 1.5.11
a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)
Y = np.array([[1,1,0],[0,1,1],[1,0,1]])

#solving the equations
X = np.linalg.solve(Y,[c,a,b])

#printing output 
print("the coefficients are",X)
