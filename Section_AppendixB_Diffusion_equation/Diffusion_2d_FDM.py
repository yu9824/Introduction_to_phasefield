# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1657695565664, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="RyuUU0-qt5rc"
import numpy as np
import matplotlib.pyplot as plt
from time import time

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1657695567711, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="EWw957i5t5re" outputId="ffa36702-6e5e-45ab-e222-ce6c1ad3a0ad"
nx, ny = 128, 128 
dx = dy = 0.5 # [m]
D = 0.1 #  [m2/s]
nsteps = 100 
dt = dx*dx*dy*dy/(2*D*(dx*dx+dy*dy))*0.5 
c0 = 1.0

# %% executionInfo={"elapsed": 422, "status": "ok", "timestamp": 1657695678841, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="X5lqWaoPt5rf"
c = np.zeros((nx, ny)) 
c_new = np.zeros((nx, ny)) 


# %% executionInfo={"elapsed": 270, "status": "ok", "timestamp": 1657695680459, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="Kyw2lJ2mt5rg"
def calc_diffusion(c, c_new):
    for j in range(ny):
        for i in range(nx):
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            if ip > nx - 1: 
                ip = nx - 1
            if im < 0:
                im = 0
            if jp > ny - 1:
                jp = ny - 1
            if jm < 0:
                jm = 0 
            c_new[i,j] = c[i,j] + D*dt*(c[ip,j] + c[im,j] +  c[i,jp]  + c[i,jm] - 4*c[i,j])/dx/dx
    c[:,:] = c_new[:,:]


# %% executionInfo={"elapsed": 506, "status": "ok", "timestamp": 1657695683374, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="za9Dh1Gct5rh"
r = 5.0 # radius of the high-concentration region
x0 = nx/2 # central potition of the high-concentration region
y0 = ny/2

for i in range(nx):
    for j in range(ny):
        r2 = (i*dx-x0*dx)**2 + (j*dy-y0*dx)**2
        if r2 < r**2:
            c[i,j] = c0

plt.imshow(c, cmap='binary')
plt.title('initial concentration')
plt.colorbar()
plt.show() 

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 55602, "status": "ok", "timestamp": 1657695740532, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="5MDjqJILt5ri" outputId="91e89d42-75d6-4952-b396-2aba0cf07f54"
start = time()
for nstep in range(nsteps+1):
    calc_diffusion(c,c_new)

    if nstep % 100 == 0:
        print('nstep = ', nstep, 'time = ', nstep*dt)
        fig = plt.figure(figsize=(7,4))
        fig.set_dpi(100)
        plt.imshow(c, cmap='binary')
        plt.title('concentration')
        plt.colorbar()
        plt.clim(0, 1) 
        #plt.show()
        plt.savefig('result{}.png'.format(nstep), format="png", dpi=300)

end = time()
print("Time for 1000 time steps =", (end-start)*1000.0, "ms")

# %% id="k7Bt_293t5rj"
