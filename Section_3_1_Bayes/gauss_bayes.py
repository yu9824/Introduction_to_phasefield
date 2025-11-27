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

# %% colab={"base_uri": "https://localhost:8080/", "height": 570} executionInfo={"elapsed": 1035, "status": "ok", "timestamp": 1674361858612, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="IKcGrJLdb6XN" outputId="76bd4246-41ca-4395-8dd5-5c3e261b6876"
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mean, var, normed=True):
    g = ((2*np.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(g)) > 0:
        g = g / sum(g)
    return g

x = np.arange(-1,1,0.01)

pdf1_mean = 0
pdf1_var = 0.02
pdf1 = gaussian(x,pdf1_mean,pdf1_var)

fig=plt.figure(dpi=150)
plt.plot(x,pdf1,color='k',label='Prior PDF: p(x)')
plt.xlabel('Variable x')
plt.ylabel('Probability density function (PDF)')
plt.xlim(-1, 1)
plt.ylim(0, 0.035)
plt.legend()
#plt.show()
plt.savefig('gauss.pdf', format="pdf", dpi=300)

# %% colab={"base_uri": "https://localhost:8080/", "height": 570} executionInfo={"elapsed": 730, "status": "ok", "timestamp": 1674361912029, "user": {"displayName": "Akinori YAMANAKA", "userId": "13314989427460547249"}, "user_tz": -540} id="hSTVpKQ9Uapt" outputId="b8446d59-20c6-414f-8247-7cb5ed1b4600"
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mean, var, normed=True):
    g = ((2*np.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(g)) > 0:
        g = g / sum(g)
    return g

def gaussian_multiply(pdf1_mean, pdf1_var, pdf2_mean, pdf2_var):
    mean = (pdf1_var * pdf2_mean + pdf2_var * pdf1_mean) / (pdf1_var + pdf2_var)
    variance = (pdf1_var * pdf2_var) / (pdf1_var + pdf2_var)
    return mean, variance

x = np.arange(-1,1,0.01)

pdf1_mean = 0.3
pdf1_var = 0.06
pdf1 = gaussian(x,pdf1_mean,pdf1_var)

pdf2_mean = -0.4
pdf2_var = 0.02
pdf2 = gaussian(x,pdf2_mean,pdf2_var)

pdf3_mean, pdf3_var = gaussian_multiply(pdf1_mean, pdf1_var, pdf2_mean, pdf2_var)
pdf3 = gaussian(x,pdf3_mean,pdf3_var)

fig=plt.figure(dpi=150)
plt.plot(x,pdf1,"--",color='k',label='Prior PDF: p(x)')
plt.plot(x,pdf2,"-.",color='k',label='Likelihood: p(y|x)')
plt.plot(x,pdf3,color='k',label='Posterior PDF: p(x|y)')
plt.xlabel('Variable x or y')
plt.ylabel('Probability density function')
plt.xlim(-1, 1)
plt.ylim(0, 0.05)
plt.legend()
#plt.show()
plt.savefig('bayes.pdf', format="pdf", dpi=300)

# %% id="NwzKqssrvzUQ"
