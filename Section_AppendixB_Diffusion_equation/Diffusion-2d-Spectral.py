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

# %% id="mKQCm9E2OYGV"
import numpy as np
import matplotlib.pyplot as plt

# %% id="ClQ2jZmJOcfY"
nx, ny = 128, 128 # number of computational grids along x and y directions
dx = dy = 0.5 # spacing of finite difference grids [m]
D = 0.3 # diffusion coefficient [m2/s]
nsteps = 10000 # number of time steps
dt = 0.01 # time increment for 1 time step
c0 = 1.0 # initial concentration in a high concentration region

# %% id="UjURKOpcOfpL"
c = np.zeros([nx, ny])
c_new = np.zeros([nx, ny])
c_k = np.zeros([nx, ny])
c_new_k = np.zeros([nx, ny])

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} executionInfo={"elapsed": 417, "status": "ok", "timestamp": 1657708375091, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="Dzy16msdOh7S" outputId="e0dfa7fc-fa43-4b04-efec-30ef04807063"
r = 5.0 # radius of the high-concentration region
x0 = nx/2 # central potition of the high-concentration region
y0 = ny/2
for i in range(nx):
    for j in range(ny):
        r2 = (i*dx-x0*dx)**2 + (j*dy-y0*dx)**2
        if r2 < r**2:
            c[i,j] = c0


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 310, "status": "ok", "timestamp": 1657709464410, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="d3ZpkYH48fH6" outputId="bb03e660-dab6-45c0-c738-8c2b962b36f5"
def calc_wave_vector(nx, ny, dx, dy):
    half_nx = int(nx/2)
    half_ny = int(ny/2)
    dkx = (2.0 * np.pi) / (nx * dx)
    dky = (2.0 * np.pi) / (ny * dy)
    k2 = np.zeros([nx, ny])
    
    for i in range(nx):
      if i < half_nx:
        kx = i*dkx
      else:
        kx = (i-nx)*dkx
      kx2 = kx**2

      for j in range(ny):
        if j < half_ny:
          ky = j*dky
        else:
          ky = (j-ny)*dky
        ky2 = ky**2

        k2[i,j] = kx2 + ky2       
    return k2

k2 = calc_wave_vector(nx, ny, dx, dy)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 15912, "status": "ok", "timestamp": 1657708390995, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="U4PPMnh5BTNw" outputId="468c575b-c722-4e56-ede6-d1387001c631"
for istep in range(nsteps+1):

  c_k = np.fft.fftn(c)
  c_new_k[:,:] = c_k[:,:] - dt * D * k2[:,:]  * c_k[:,:] 

  c = np.real(np.fft.ifftn(c_new_k))

  if istep % 1000 == 0:
    print('nstep = ', istep, 'time = ', istep*dt)
    plt.imshow(c, cmap='bwr')
    plt.title('concentration of B atom')
    plt.colorbar()
    plt.show() 

# %% id="b5duaKZfPTXl"
