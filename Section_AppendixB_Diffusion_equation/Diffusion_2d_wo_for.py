# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1644212608966, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="pOr0jCHMuGLK"
import numpy as np
import matplotlib.pyplot as plt
from time import time

# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1644212609329, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="qu0EyrAYuGLN"
nx, ny = 128, 128 # number of computational grids along x and y directions
dx = dy = 0.5 # spacing of finite difference grids [m]
D = 0.3 # diffusion coefficient [m2/s]
nsteps = 1000 # number of time steps
dt = dx*dx*dy*dy/(2*D*(dx*dx+dy*dy))*0.5 # time increment for 1 time step
c0 = 1.0 # initial concentration in a high concentration region

# %% executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1644212609329, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="Ws9GDC19uGLP"
c = np.zeros((nx, ny)) # array for a concentration-fieldat time t 
c_new = np.zeros((nx, ny)) # array for a concentration-fieldat time t+dt


# %% executionInfo={"elapsed": 31, "status": "ok", "timestamp": 1644212609330, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="iXkJCOKluGLR"
def calc_diffusion(c, c_new):
    c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + D*dt* (c[2:, 1:-1] + c[:-2, 1:-1] + c[1:-1, 2:]  + c[1:-1, :-2] - 4*c[1:-1, 1:-1]) /dx/dx  

    c_new[0,:] = c_new[1,:]
    c_new[nx-1,:] = c_new[nx-2,:]
    c_new[:,0] = c_new[:,1]
    c_new[:,ny-1] = c_new[:,ny-2]
    c[:,:] = c_new[:,:]


# %% executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1644212609331, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="MV2HPdC_uGLS"
r = 5.0 # radius of the high-concentration region
x0 = nx/2 # central potition of the high-concentration region
y0 = ny/2

for i in range(nx):
    for j in range(ny):
        r2 = (i*dx-x0*dx)**2 + (j*dy-y0*dx)**2
        if r2 < r**2:
            c[i,j] = c0

plt.imshow(c, cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show() 

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 3591, "status": "ok", "timestamp": 1644212612891, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="Cn_snGmguGLT" outputId="f966cf7f-0875-490e-d5cc-e44714b0d41c"
start = time()
for nstep in range(nsteps+1):
    calc_diffusion(c,c_new)

    if nstep % 100 == 0:
        print('nstep = ', nstep)
        plt.imshow(c, cmap='bwr')
        plt.title('concentration')
        plt.colorbar()
        plt.show() 

end = time()
print("Time for 1000 time steps =", (end-start)*1000.0, "ms")
