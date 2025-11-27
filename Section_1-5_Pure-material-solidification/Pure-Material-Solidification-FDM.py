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

# %% id="vftjMEXQkI9R"
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from time import time

# %% id="mXbqgVPdkLmX"
nx = ny = 256 # 差分格子点数
dx = dy = 2.0e-08 # 差分格子点の間隔[m]
dt = 5.0e-12 # 時間増分[s]
stepmax = 7000 # 時間ステップの総数
pi = np.pi 
delta = 4.0 * dx # 界面幅[m]
gamma = 0.37 # 界面エネルギー [J/m2]
zeta = 0.03 # 異方性強度
aniso = 4.0 # 異方性モード数
angle0 = 0.*pi # 優先成長方向
T_melt = 1728.0 # 融点Tm [K]
K = 84.01 # 熱伝導率 [W/(mK)]
c = 5.42e+06 # 比熱 [J/K]
latent = 2.35e+09 # 潜熱 [J/mol]
lamb = 0.1
b = 2.0 * np.arctanh(1.0-2.0*lamb)
mu = 2.0 # 界面カイネティック係数 [m/(Ks)]
kappa = K / c # 熱拡散係数
a0 = np.sqrt(3.0*delta*gamma/b) # 勾配エネルギー係数 
www = 6.0 * gamma * b / delta # エネルギー障壁の高さ
pmobi = b * T_melt * mu / (3.0 * delta * latent) # フェーズフィールドモビリティー 
T_0 = 1424.5 # 系の温度 [K]

# %% id="bL1tYLoakYJc"
phi = np.zeros((nx,ny)) # 時刻tでのフェーズフィールド変数
phi_new = np.zeros((nx,ny)) # 時刻t+dtでのフェーズフィールド変数
temp = np.zeros((nx,ny)) # 時刻tでの温度
temp_new = np.zeros((nx,ny)) # 時刻t+dtでの温度
grad_phix = np.zeros((nx,ny)) # x方向のフェーズフィールド変数の勾配 
grad_phiy = np.zeros((nx,ny)) # y方向のフェーズフィールド変数の勾配 
a2 = np.zeros((nx,ny)) # (勾配エネルギー係数)^2
lap_temp = np.zeros((nx,ny)) # 温度のラプラシアン
lap_phi = np.zeros((nx,ny)) # フェーズフィールドのラプラシアン
ax = np.zeros((nx,ny)) 
ay = np.zeros((nx,ny))

# %% id="2jMWkmX4kjyg"
r0 = 3.*dx
for j in range(0,ny):
    for i in range(0,nx):
        phi[i,j] = 0.0
        x = dx*(i-nx/2)
        y = dy*(j-ny/2)
        r = np.sqrt(x*x + y*y)
        phi[i,j] = 0.5*(1.-np.tanh(np.sqrt(2.*www)/(2.*a0)*(r-r0))) # フェーズフィールド変数の初期分布
        if phi[i,j] <= 1.0e-5:
            phi[i,j] = 0.0
        temp[i,j] = T_0 + phi[i,j] * (T_melt-T_0) # 温度の初期分布


# %% id="K-ANeSeskbob"
@jit
def calcgrad(phi,temp,zeta,a0,www,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,a2):
    for j in range(ny):
        for i in range(nx):
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            if ip > nx-1:
                ip = nx - 1
            if im < 0:
                im = 0
            if jp > ny-1:
                jp = ny - 1
            if jm < 0:
                jm = 0

            grad_phix[i,j] = (phi[ip,j]-phi[im,j])/(2.*dx)
            grad_phiy[i,j] = (phi[i,jp]-phi[i,jm])/(2.*dy)
            lap_phi[i,j] = (2.*(phi[ip,j]+phi[im,j]+phi[i,jp]+phi[i,jm])+phi[ip,jp]+phi[im,jm]+phi[im,jp]+phi[ip,jm]-12.*phi[i,j])/(3.*dx*dx)
            lap_temp[i,j]= (2.*(temp[ip,j]+temp[im,j]+temp[i,jp]+temp[i,jm])+temp[ip,jp]+temp[im,jm]+temp[im,jp]+temp[ip,jm]-12.*temp[i,j])/(3.*dx*dx)

            if grad_phix[i,j] == 0.:
                if grad_phiy[i,j] > 0.:
                    angle = 0.5*pi
                else:
                    angle = -0.5*pi
            elif grad_phix[i,j] > 0.:
                if grad_phiy[i,j] > 0.:
                    angle = np.arctan(grad_phiy[i,j]/grad_phix[i,j])
                else:
                    angle = 2.0*pi + np.arctan(grad_phiy[i,j]/grad_phix[i,j])
            else:
                angle = pi + np.arctan(grad_phiy[i,j]/grad_phix[i,j])

            a = a0*(1. + zeta * np.cos(aniso*(angle-angle0)))
            dadtheta = -a0*aniso*zeta*np.sin(aniso*(angle-angle0))
            ay[i,j] = -a * dadtheta * grad_phiy[i,j]
            ax[i,j] =  a * dadtheta * grad_phix[i,j]
            a2[i,j] = a * a


# %% id="vh2bLbqLkgSd"
@jit
def timeevol(phi,temp,zeta,a0,www,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,a2,phi_new,temp_new):
    for j in range(ny):
        for i in range(nx):
            ip = i + 1
            im = i - 1
            jp = j + 1
            jm = j - 1
            if ip > nx-1:
                ip = nx - 1
            if im < 0:
                im = 0
            if jp > ny-1:
                jp = ny -1
            if jm < 0:
                jm = 0

            dxdy = (ay[ip,j]-ay[im,j])/(2.*dx)
            dydx = (ax[i,jp]-ax[i,jm])/(2.*dy)
            grad_a2x = (a2[ip,j]-a2[im,j])/(2.*dx)
            grad_a2y = (a2[i,jp]-a2[i,jm])/(2.*dy)
            tet = phi[i,j]
            drive = -latent * (temp[i,j]-T_melt) / T_melt
            scal = grad_a2x*grad_phix[i,j]+grad_a2y*grad_phiy[i,j]

            phi_new[i,j] = phi[i,j] + (dxdy + dydx + a2[i,j]*lap_phi[i,j] + scal + 4.0*www*tet*(1.0-tet)*(tet-0.5+15.0/(2.0*www)*drive*tet*(1.0-tet)))*dt*pmobi
            temp_new[i,j] = temp[i,j] + kappa*lap_temp[i,j]*dt + 30.0*tet*tet*(1.0-tet)*(1.0-tet)*(latent/c)*(phi_new[i,j]-tet)


# %% colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"elapsed": 120117, "status": "ok", "timestamp": 1666438502493, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="KS3KCo9LkoSf" outputId="5694a012-b426-4531-8064-bf6cc157ebfd"
for nstep in range(stepmax):
    calcgrad(phi,temp,zeta,a0,www,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,a2)
    timeevol(phi,temp,zeta,a0,www,grad_phix,grad_phiy,lap_phi,lap_temp,ax,ay,a2,phi_new,temp_new)
    phi = phi_new
    temp = temp_new
    
    if nstep % 700 == 0:
        plt.figure(figsize=(12,6))
        plt.rcParams["font.size"] = 15
        plt.subplot(121)
        plt.imshow(phi, cmap="bwr")
        plt.title('Phase-field')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(temp, cmap="bwr")
        plt.title('Temperature [K]')
        plt.colorbar()
        plt.show()
