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

# %% executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1666438022704, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="nM1rUxnhuWkv"
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# %% colab={"base_uri": "https://localhost:8080/", "height": 334} executionInfo={"elapsed": 366, "status": "ok", "timestamp": 1666438023062, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="ZmW-B0OwuoH_" outputId="860367d6-1921-4466-f9ff-48fabd15ed94"
nx = ny = 128 # 差分格子点数
dx = dy = 1.0 # 差分格子点の間隔
total_step = 1000 # 時間ステップの総数
dt = 1.0e-2 # 時間増分
A = 2.0 # 化学的自由エネルギーの係数A  
mobility = 1.0 # 拡散モビリティ
grad_coef = 1.0 # 濃度勾配エネルギー係数
c0 = 0.5 # 初期平均濃度

fig = plt.figure(figsize=(5,5))
cc = np.linspace(0.01, 0.99, 100);
plt.plot(cc, cc**2*(1.-cc)**2 ,color='black')
plt.plot(c0, c0**2*(1.-c0)**2 ,color='r',marker='o',markersize=10)
plt.xlabel('Concentration c [at. frac]')
plt.ylabel('Chemical free energy density')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} executionInfo={"elapsed": 620, "status": "ok", "timestamp": 1666438023679, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="ZsRQ8VPRu4Gb" outputId="2e8bc829-dd44-4b2e-8896-280c6fe2aae0"
con = np.zeros([nx, ny]) # 時刻tでの濃度変数
con_new = np.zeros([nx, ny]) # 時刻t+dtでの濃度変数
con = c0 + 0.01 * (0.5 - np.random.rand(nx, ny)) # 初期濃度場の設定

plt.imshow(con, cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show() 


# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1666438023679, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="e3UHIU-YvCTY"
@jit
def update(con, con_new):
  for j in range(ny):
    for i in range(nx):
      
      ip = i + 1
      im = i - 1
      jp = j + 1
      jm = j - 1
      ipp = i + 2
      imm = i - 2
      jpp = j + 2
      jmm = j - 2

      if ip > nx-1:
        ip = ip - nx
      if im < 0:
        im = im + nx
      if jp > ny-1:
        jp = jp - ny
      if jm < 0:
        jm = jm + ny
      if ipp > nx-1:
        ipp = ipp - nx
      if imm < 0:
        imm = imm + nx
      if jpp > ny-1:
        jpp = jpp - ny
      if jmm < 0:
        jmm = jmm + ny
      
      cc = con[i,j] 
      ce = con[ip,j] 
      cw = con[im,j] 
      cs = con[i,jm] 
      cn = con[i,jp] 
      cse = con[ip,jm]
      cne = con[ip,jp]
      csw = con[im,jm]
      cnw = con[im,jp]
      cee = con[ipp,j]  
      cww = con[imm,j]
      css = con[i,jmm]
      cnn = con[i,jpp]
      
      mu_chem_c = 2.*A*cc*(1.-cc)**2 - 2.*A*cc**2*(1.-cc) 
      mu_chem_w = 2.*A*cw*(1.-cw)**2 - 2.*A*cw**2*(1.-cw)  
      mu_chem_e = 2.*A*ce*(1.-ce)**2 - 2.*A*ce**2*(1.-ce) 
      mu_chem_n = 2.*A*cn*(1.-cn)**2 - 2.*A*cn**2*(1.-cn) 
      mu_chem_s = 2.*A*cs*(1.-cs)**2 - 2.*A*cs**2*(1.-cs)  
      
      mu_grad_c = -grad_coef*( (ce - 2.0*cc + cw)/dx/dx + (cn  - 2.0*cc + cs)/dy/dy) 
      mu_grad_w = -grad_coef*( (cc - 2.0*cw + cww)/dx/dx + (cnw - 2.0*cw + csw)/dy/dy)
      mu_grad_e = -grad_coef*( (cee - 2.0*ce + cc)/dx/dx + (cne - 2.0*ce + cse)/dy/dy)  
      mu_grad_n = -grad_coef*( (cne - 2.0*cn + cnw)/dx/dx + (cnn - 2.0*cn + cc)/dy/dy) 
      mu_grad_s = -grad_coef*( (cse - 2.0*cs + csw)/dx/dx + (cc  - 2.0*cs + css)/dy/dy)
      
      mu_c = mu_chem_c + mu_grad_c 
      mu_w = mu_chem_w + mu_grad_w 
      mu_e = mu_chem_e + mu_grad_e 
      mu_n = mu_chem_n + mu_grad_n 
      mu_s = mu_chem_s + mu_grad_s
      
      laplace_mu = (mu_w - 2.0*mu_c + mu_e)/dx/dx + (mu_n - 2.0 *mu_c + mu_s)/dy/dy # 拡散ポテンシャルの2階微分
      con_new[i,j] = con[i,j] + mobility*laplace_mu*dt # カーン-ヒリアード方程式の計算   


# %% colab={"base_uri": "https://localhost:8080/", "height": 299} executionInfo={"elapsed": 62483, "status": "ok", "timestamp": 1666438086144, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="VmYAeWUxgZTP" outputId="9c8ef00f-6335-4c08-c3b4-88c3789b0001"
for nstep in range(total_step):
    update(con, con_new)
    con[:,:] = con_new[:,:]
    
    if nstep % 100 == 0:
        print('nstep = ', nstep)
        print('Maximum concentration = ', np.max(con))
        print('Minimum concentration = ', np.min(con))
        plt.figure(figsize=(6,6))
        plt.imshow(con, cmap='bwr')
        plt.title('concentration of B atom')
        plt.colorbar()
        plt.show() 

# %% executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1666438086144, "user": {"displayName": "Akinori Yamanaka", "userId": "04881339994405091902"}, "user_tz": -540} id="EOPhZQ9_m5LY"
