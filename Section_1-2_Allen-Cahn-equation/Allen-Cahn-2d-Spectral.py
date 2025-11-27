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
# This program is under GNU GENERAL PUBLIC LICENSE Version 2, June 1991. See https://www.gnu.org/licenses/old-licenses/gpl-2.0.ja.html for the detail.

# %% executionInfo={"elapsed": 460, "status": "ok", "timestamp": 1657710786665, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="RloewLCM_u_l"
import numpy as np
import matplotlib.pyplot as plt

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1657710787056, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="p5T6NSJ2ARcX"
nx, ny = 32, 32 
dx, dy = 0.5e-6, 0.5e-6
eee = 1.0e+6
gamma = 1.0 
delta = 4.*dx 
amobi = 4.e-14
ram = 0.1 
bbb = 2.*np.log((1.+(1.-2.*ram))/(1.-(1.-2.*ram)))/2. 

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1657710787057, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="05pZP3XrAR-P"
aaa   = np.sqrt(3.*delta*gamma/bbb) 
www   = 6.*gamma*bbb/delta 
pmobi = amobi*np.sqrt(2.*www)/(6.*aaa) 

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1657710787058, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="bdfQ_VntAT_J"
dt = 0.1
nsteps = 1000 

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1657710787058, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="WpBwyHjXAVhM"
p  = np.zeros((nx,ny)) 
dfdp = np.zeros([nx, ny])

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} executionInfo={"elapsed": 544, "status": "ok", "timestamp": 1657710787596, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="5ZoZbf5NAXKg" outputId="daee5668-d175-4002-ade5-180d6baeffa7"
r_nuclei = 5.*dx 
for i in range(nx):
    for j in range(ny):
        r = np.sqrt( ((i-nx/2) *dx)**2 +((j-ny/2) * dy)**2 ) - r_nuclei
        p[i,j] = 0.5*(1.-np.tanh(np.sqrt(2.*www)/(2.*aaa)*r))


# %% executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1657710787597, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="rj-6W6R6AoLU"
# 波数ベクトルkとk^2を計算するための関数
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


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1657710787598, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="KGBp4Im4R00r" outputId="023f9e55-c61b-4600-a00b-3a2fa896da44"
k2 = calc_wave_vector(nx, ny, dx, dy)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 3004, "status": "ok", "timestamp": 1657710790590, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="68xLFvJHAZDP" outputId="ad08f93c-65a2-4cc0-d51f-97c5d4e62191"
for istep in range(1, nsteps):

  pk = np.fft.fftn(p)
  dfdp[:,:] = 4.0 * www * p[:,:]*(1 - p[:,:])*(p[:,:] - 0.5 + 3.0/2.0/www * eee) 

  dfdpk = np.fft.fftn(dfdp)
 
  nummer = pmobi * dfdpk[:,:] * dt
  denom = 1.0 + pmobi * k2[:,:] * aaa*aaa * dt
  pk[:,:] = (pk[:,:] + nummer) / denom

  p = np.real(np.fft.ifftn(pk))

  for i in range(1, nx):
    for j in range(1, ny):
      if(p[i, j] >= 0.9999):
        p[i, j] = 0.9999
      if(p[i, j] < 0.00001):
        p[i, j] = 0.00001

  if istep % 100 == 0:
      print('nstep = ', istep)
      plt.figure(figsize=(6,6))
      plt.imshow(p, cmap='bwr')        
      plt.title('phase-field')
      plt.colorbar()
      plt.show() 

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1657710790591, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="OVZi8eN_U-uQ"
