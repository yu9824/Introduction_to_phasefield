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

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 129202, "status": "ok", "timestamp": 1644243394768, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="RyuUU0-qt5rc" outputId="55127a16-a11a-404f-d524-84da99c08461"
# %matplotlib nbagg
import numpy as np 
import matplotlib.pyplot as plt

# !pip install pycuda # install pycuda
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1644243394769, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="EWw957i5t5re"
nx, ny = 128, 128 # number of computational grids along x and y directions
dx = dy = 0.5 # spacing of finite difference grids [m]
D = 0.3 # diffusion coefficient [m2/s]
nsteps = 1000 # number of time steps
dt = dx*dx*dy*dy/(2*D*(dx*dx+dy*dy))*0.5 # time increment for 1 time step
c0 = 1.0 # initial concentration in a high concentration region

# %% executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1644243394770, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="X5lqWaoPt5rf"
c = np.zeros((nx, ny)).astype(np.float32) # array for a concentration-fieldat time t 
c_new = np.zeros((nx, ny)).astype(np.float32) # array for a concentration-fieldat time t+dt


# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1644243394771, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="DfbBUdsnXq6h"
# CUDA Cで記述する
def get_kernel_string(nx, ny, dx, dy, dt, D):
    return """
    #define nx %d
    #define ny %d
    #define dx %f
    #define dy %f
    #define dt %f
    #define D  %f
    __global__ void diffuse_kernel(float *u_new, float *u) {

        int i = blockIdx.x * block_size_x + threadIdx.x;
        int j = blockIdx.y * block_size_y + threadIdx.y;

        int ip = i + 1;
        int im = i - 1;
        int jp = j + 1;
        int jm = j - 1;
        if(ip > nx-1) { ip = nx - 1; }
        if(im < 0  ) { im = 0; }
        if(jp > ny-1) { jp = ny - 1;}
        if(jm < 0  ) { jm = 0; }

        u_new[j*nx+i] = u[j*nx+i] + D * ( (u[j*nx+ ip] - 2.0f*u[j*nx+i] + u[j*nx+ im])/dx/dx + ( u[( jp )*nx+i] - 2.0f*u[j*nx+i] + u[( jm )*nx+i] )/dy/dy )*dt;
    }
    """ % (nx, ny, dx, dy, dt, D)

kernel_string = get_kernel_string(nx, ny, dx, dy, dt, D)

# %% colab={"base_uri": "https://localhost:8080/", "height": 281} executionInfo={"elapsed": 554, "status": "ok", "timestamp": 1644243395308, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="za9Dh1Gct5rh" outputId="b5d10566-6ec8-4cff-87b3-e9d9d2243a9a"
r = 5.0 # radius of the high-concentration region
x0 = nx/2 # central potition of the high-concentration region
y0 = ny/2

for i in range(nx):
    for j in range(ny):
        r2 = (i*dx-x0*dx)**2 + (j*dy-y0*dx)**2
        if r2 < r**2:
            c[i,j] = c0

plt.imshow(c, cmap='bwr')
plt.title('concentration')
plt.colorbar()
plt.show() 

# %% colab={"base_uri": "https://localhost:8080/", "height": 370} executionInfo={"elapsed": 1567, "status": "ok", "timestamp": 1644243396850, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="6ePbhKCWYpKm" outputId="d3e88b68-0362-49da-e80a-152f14589a4e"
#initialize PyCuda and get compute capability needed for compilation
drv.init()
context = drv.Device(0).make_context()
devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

#allocate GPU memory
u_old = drv.mem_alloc(c.nbytes)
u_new = drv.mem_alloc(c_new.nbytes)

#setup thread block dimensions and compile the kernel
threads = (16,16,1)
grid = (int(nx/16), int(ny/16), 1)

block_size_string = "#define block_size_x 16\n#define block_size_y 16\n"
diffuse_kernel = SourceModule(block_size_string+kernel_string, arch='sm_'+cc).get_function("diffuse_kernel")

#create events for measuring performance
start = drv.Event()
end = drv.Event()

#move the data to the GPU
drv.memcpy_htod(u_old, c)
drv.memcpy_htod(u_new, c)

#call the GPU kernel a 1000 times and measure performance
context.synchronize()
start.record()
for i in range(500):
    diffuse_kernel(u_new, u_old, block=threads, grid=grid)
    diffuse_kernel(u_old, u_new, block=threads, grid=grid)
end.record()
context.synchronize()
print("1000 steps of diffuse took", end.time_since(start), "ms.")

# #copy the result from the GPU to Python for plotting
gpu_result = np.zeros_like(c)
drv.memcpy_dtoh(gpu_result, u_new)

plt.imshow(gpu_result, cmap='bwr')
plt.title('concentration')
plt.colorbar()
plt.show() 

# %% executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1644243396852, "user": {"displayName": "Akinori YAMANAKA", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjH-a8vAX5pdveRaEOJqo-g9_KWgbs3uJ4r0GXUpA=s64", "userId": "13314989427460547249"}, "user_tz": -540} id="wytPzkYFxVir"
