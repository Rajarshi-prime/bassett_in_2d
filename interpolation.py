'''
title : interpolation.py
@author : Rajarshi + Divya (later)
Comments : testing interpolation of velocity and the gradient
'''

#! Todo:
#* Make the non-dimensionalization such that things change accordingly with the Stokes number. 
#* Decrease the Stokes number. 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
import sys
from ndsplines import make_interp_spline,from_file
from time import time



method = 'cubic'

print("Method: ", method)

# code = str(sys.argv[-2]) #? options are omg and str
code = "rndm"
init = "zero"
# init = "fluid"

print(f'code : {code}')

## ------------------ Params ---------------------
d = 2 # Dimension of the system
Nprtcl = 2 #! Number of particles 
# prtcl_idx = int(sys.argv[-1]) #! For different serial loops
PI = np.pi
TWO_PI = 2*PI



## -------------- Making the Grid ----------------
#? The field is slabbed along Z.

N = 128
L = TWO_PI #! As the turbulence data is periodic in 1024 points, it is translation invariant. 
X = Y = np.linspace(0, L, N+1, endpoint= True)
# X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy = X[1]-X[0], Y[1]-Y[0]

## -----------------------------------------------
print("Grid Done")





#* ------------- Velocity field +Interpolation ------------------
loadPath = pathlib.Path(f"D:/Nextcloud/Fluids/bassett_in_2d/data") #! Change this accordingly
veldataname = loadPath/f"interpVel_{N}.npz"
Ainterpname = loadPath/f"interpA_{N}.npz"
# ugrdAinterpname = loadPath/f"interpugrdA_{N}.npz"
notRewrite = True
if veldataname.exists() and notRewrite and Ainterpname.exists():
    print("Exists")
    uinterp = from_file(veldataname)
    Ainterp = from_file(Ainterpname)
    
else:
    xg, yg = np.meshgrid(X, Y, indexing='ij')
    print("Doesn't exist")
    
    # u_field = np.load(loadPath/f"vel_{N}_xr_{xrank}_yr_{yrank}_zr_{zrank}.npz")["field"]    
    
    udat = np.load(loadPath/f"vel_{N}.npz")["field"]    
    u_field = np.zeros((d,N+1,N+1))
    u_field[:,:-1,:-1] = udat
    del udat
    u_field[:,:-1,-1] = u_field[:,:-1,0]
    u_field[:,-1,:] = u_field[:,0,:]
    
    print(np.allclose(u_field[:,0],u_field[:,-1]),"for x direction")
    print(np.allclose(u_field[:,:,0],u_field[:,:,-1]),"for y direction")
    # print(np.allclose(u_field[:,:,:,0],u_field[:,:,:,-1]),"for z direction")
    
    
    # print(f"Maximum dimensional Velocity: {np.max(np.abs(u_field)*TWO_PI)}")
    
    uinterp = make_interp_spline(np.stack((xg,yg),axis = -1),np.moveaxis(u_field,[0],[2]))
    uinterp._extrapolate = uinterp._extrapolate*False
    uinterp.to_file(veldataname)
    del u_field,uinterp
    
    # A_field = np.load(loadPath/f"shear_{N}_xr_{xrank}_yr_{yrank}_zr_{zrank}.npz")["shear"]
    A_dat = np.load(loadPath/f"shear_{N}.npz")["shear"]
    A_field = np.zeros((d,d,N+1,N+1))
    A_field[...,:-1,:-1] = A_dat
    del A_dat
    A_field[...,:-1,-1] = A_field[...,:-1,0]
    A_field[...,-1,:] = A_field[...,0,:]
    
    print(np.allclose(A_field[...,0,],A_field[...,-1]),"for x direction")
    print(np.allclose(A_field[...,0],A_field[...,-1]),"for y direction")
    # print(np.allclose(A_field[...,0],A_field[...,-1]),"for z direction")
    
    print("Max trace:", np.max(np.abs(np.einsum('ii...->...',A_field))))
    print("Min value:", np.min(np.abs(A_field)))
    Ainterp = make_interp_spline(np.stack((xg,yg),axis = -1),np.moveaxis(A_field,[0,1],[2,3]))
    Ainterp._extrapolate = Ainterp._extrapolate*False
    Ainterp.to_file(Ainterpname)
    
    del A_field,Ainterp,xg,yg
    
    
    uinterp = from_file(veldataname)
    Ainterp = from_file(Ainterpname)
    


#* ---------------------------------------------------------------
    
xp = np.random.uniform(0,L,(Nprtcl,d))
u = uinterp(xp%L)
A = Ainterp(xp%L)
print(f"Particles are at",xp)
print(f"Velocity at particles",u)
print(f"Gradient at particles",A)