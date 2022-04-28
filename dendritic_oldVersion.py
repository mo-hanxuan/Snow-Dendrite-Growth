"""
    Here the phase field model of dendritic solidification refer to this thesis:
        Kobayashi, R. (1993), "Modeling and numerical simulations of dendritic crystal growth." 
        Physica D 63(3-4): 410-423
    
    this version is directly translated from the old version of a matlab-code
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n = 400
phi = ti.field(dtype=ti.float64, shape=(n, n))
phiNew = ti.field(dtype=ti.float64, shape=(n, n))
tp = ti.field(dtype=ti.float64, shape=(n, n))  # temperature
tpNew = ti.field(dtype=ti.float64, shape=(n, n))
dEnergy_dGrad = ti.Vector.field(2, ti.float64, shape=(n, n))  # deritave of energy with respect to gradient of phi
epsilon = ti.field(dtype=ti.float64, shape=(n, n))
epsilon_derivative = ti.field(dtype=ti.float64, shape=(n, n))
grad_phi = ti.Vector.field(2, ti.float64, shape=(n, n))
lap_phi = ti.field(dtype=ti.float64, shape=(n, n))
lap_tp = ti.field(dtype=ti.float64, shape=(n, n))
angles = ti.field(dtype=ti.float64, shape=(n, n))
grad_epsilon2 = ti.Vector.field(2, ti.float64, shape=(n, n))

dx = 0.03
dt = 3.e-4
tau = 3.e-4

epsilonbar = 0.005  # 0.005 gradient energy coefficient
mu = 1.0
k = 1.5   # 1.5 latent heat coefficient
delta = 0.12  # 0.02 the strength of anisotropy
anisoMod = 6.   # mode number of anisotropy
alpha = 0.9 / np.pi  # 0.9 
gamma = 10.0
teq = 1.0  # temperature of equilibrium
mo = 1. / tau  # mobility
angle0 = np.pi / 18. * 1.5


@ti.func
def sumVec(vec):
    res = 0.
    for i in ti.static(range(vec.n)):
        res += vec[i]
    return res


@ti.kernel
def initializeVariables():
    radius = 1.
    center = ti.Vector([n//2, n//2])
    for i, j in phi:
        if sumVec((ti.Vector([i, j]) - center)**2) < radius**2:
            phi[i, j] = 1.
        else:
            phi[i, j] = 0.
        tp[i, j] = 0.  # temperature


@ti.func
def neighbor_index(i, j):
    """
        use periodic boundary condition to get neighbor index
    """
    im = i - 1 if i - 1 >= 0 else n - 1
    jm = j - 1 if j - 1 >= 0 else n - 1
    ip = i + 1 if i + 1 < n else 0
    jp = j + 1 if j + 1 < n else 0
    return im, jm, ip, jp


@ti.func
def laplacian_phi(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    # return (phi[ip, j] + phi[im, j] - 2 * phi[i, j]) / dx**2 + \
    #        (phi[i, jp] + phi[i, jm] - 2 * phi[i, j]) / dx**2
    return (
        2 * (phi[im, j] + phi[i, jm] + phi[ip, j] + phi[i, jp]) 
        + (phi[im, jm] + phi[im, jp] + phi[ip, jm] + phi[ip, jp]) 
        - 12 * phi[i, j]
    ) / (3. * dx * dx)


@ti.func
def laplacian_temperature(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    # return (tp[ip, j] + tp[im, j] - 2 * tp[i, j]) / dx**2 + \
    #        (tp[i, jp] + tp[i, jm] - 2 * tp[i, j]) / dx**2
    return (
        2 * (tp[im, j] + tp[i, jm] + tp[ip, j] + tp[i, jp]) 
        + (tp[im, jm] + tp[im, jp] + tp[ip, jm] + tp[ip, jp]) 
        - 12 * tp[i, j]
    ) / (3. * dx * dx)


@ti.func
def gradient(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    return ti.Vector([
        (phi[ip, j] - phi[im, j]) / (2. * dx),
        (phi[i, jp] - phi[i, jm]) / (2. * dx)
    ])


@ti.func
def divergence_dEnergy_dGrad(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    return (dEnergy_dGrad[ip, j][0] - dEnergy_dGrad[im, j][0]) / (2. * dx) + \
           (dEnergy_dGrad[i, jp][1] - dEnergy_dGrad[i, jm][1]) / (2. * dx)


@ti.kernel
def get_gradient_and_laplacian():
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)
        grad_phi[i, j] = gradient(i, j)

        lap_phi[i, j] = laplacian_phi(i, j)
        lap_tp[i, j] = laplacian_temperature(i, j)

        angles[i, j] = ti.atan2(grad_phi[i, j][1], grad_phi[i, j][0])
        epsilon[i, j] = epsilonbar * (1. + delta * ti.cos(anisoMod * (angles[i, j] - angle0)))
        epsilon_derivative[i, j] = -epsilonbar * anisoMod * delta * ti.sin(anisoMod * (angles[i, j] - angle0))
        
        grad_epsilon2[i, j][0] = (epsilon[ip, j]**2 - epsilon[im, j]**2) / dx
        grad_epsilon2[i, j][1] = (epsilon[i, jp]**2 - epsilon[i, jm]**2) / dx


@ti.kernel
def evolution():
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)
        term1 = (
            epsilon[i, jp] * epsilon_derivative[i, jp] * grad_phi[i, jp][0] - \
            epsilon[i, jm] * epsilon_derivative[i, jm] * grad_phi[i, jm][0]
        ) / (2. * dx)
        term2 = -(
            epsilon[ip, j] * epsilon_derivative[ip, j] * grad_phi[ip, j][1] - \
            epsilon[im, j] * epsilon_derivative[im, j] * grad_phi[im, j][1] 
        ) / (2. * dx)
        term3 = grad_epsilon2[i, j][0] * grad_phi[i, j][0] + \
                grad_epsilon2[i, j][1] * grad_phi[i, j][1]

        phiOld = phi[i, j]
        m = alpha * ti.atan2(gamma * (teq - tp[i, j]), 1.)

        ### time evolution
        phiNew[i, j] = phi[i, j] + (
            term1 + term2 + \
            term3 + epsilon[i, j]**2 * lap_phi[i, j] + \
            phiOld * (1.0 - phiOld) * (phiOld - 0.5 + m)
        ) * dt / tau
        tpNew[i, j] = tp[i, j] + lap_tp[i, j] * dt + k * (phiNew[i, j] - phiOld)


@ti.kernel
def updateVariables():
    for i, j in phi:
        phi[i, j] = phiNew[i, j]
        tp[i, j] = tpNew[i, j]


if __name__ == "__main__":
    initializeVariables()
    gui1 = ti.GUI("phase field", res=(n, n))
    gui2 = ti.GUI("temperature field", res=(n, n))

    for i in range(1000000):
        gui1.set_image(phi)
        gui1.show()
        get_gradient_and_laplacian()
        evolution()
        updateVariables()
        gui2.set_image(tp)
        gui2.show()

