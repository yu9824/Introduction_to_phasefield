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

# %% [markdown]
# This program is under GNU GENERAL PUBLIC LICENSE Version 2, June 1991. 
# See https://www.gnu.org/licenses/old-licenses/gpl-2.0.ja.html for the detail. 

# %% id="hCb8t76LFN_o"
import numpy as np
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 334} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1666793562241, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="gqiQDxIXXrfc" outputId="88d3206f-a6cc-429e-85ec-9612a3ae1813"
nx = ny = 128
dx = dy = 1.0
total_step = 1000
dt = 1.0e-2
A = 2.0
mobility = 1.0
grad_coef = 1.0
c0 = 0.5

fig = plt.figure(figsize=(5,5))
cc = np.linspace(0.01, 0.99, 100);
plt.plot(cc, cc**2*(1.-cc)**2 ,color='black')
plt.plot(c0, c0**2*(1.-c0)**2 ,color='r',marker='o',markersize=10)
plt.xlabel('Concentration c')
plt.ylabel('Chemical free energy density')
plt.show()

# %% id="F5htgiCWPpXB"
con = np.zeros([nx, ny]) 
dfdcon = np.zeros([nx, ny]) 
con = c0 + 0.01 * (0.5 - np.random.rand(nx, ny))

plt.imshow(con, cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show() 


# %% id="Yd5keFoVPWmh"
def calc_wave_vector(nx, ny, dx, dy):
	nx21 = int(nx/2 + 1)
	ny21 = int(ny/2 + 1)
	nx2 = nx + 2
	ny2 = ny + 2
	dkx = (2.0 * np.pi) / (nx * dx)
	dky = (2.0 * np.pi) / (ny * dy)
	kx = np.zeros([nx, ny])
	ky = np.zeros([nx, ny])
	k2 = np.zeros([nx, ny])
	k4 = np.zeros([nx, ny])

	for i in range(1, nx21):
		for j in range(1, ny):
			fk1 = (i - 1) * dkx
			kx[i, :] = fk1
			kx[nx - i, :] = -fk1
	for i in range(1, nx):
		for j in range(1, ny21):
			fk2 = (j - 1) * dky
			ky[:, j] = fk2
			ky[:, ny - j] = -fk2

	k2[:, :] = kx[:, :]**2 + ky[:, :]**2
	return k2

k2 = calc_wave_vector(nx, ny, dx, dy)
k4 = k2 **2

# %% colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"elapsed": 24867, "status": "ok", "timestamp": 1666793588343, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="yKMjiqxyPvqh" outputId="a4d4e753-6dc8-483c-d8c5-f80190fac211"
for istep in range(total_step+1):
    conk = np.fft.fftn(con) 
    dfdcon = (2.0 * con * (1 - con)**2 -2.0 * con**2 * (1 - con)) 
    dfdconk = np.fft.fftn(dfdcon)  
    nummer = dt * mobility * A * k2 * dfdconk
    denom = 1.0 + dt * mobility * grad_coef * k4
    conk = (conk - nummer) / denom 
    con = np.real(np.fft.ifftn(conk)) 
    
    if istep % 100 == 0:
        print('nstep = ', istep)
        print('Maximum concentration = ', np.max(con))
        print('Minimum concentration = ', np.min(con))
        plt.imshow(con, cmap='bwr')
        plt.title('concentration of B atom')
        plt.colorbar()
        plt.show() 

# %% id="LJkUC2BqQHRo"
