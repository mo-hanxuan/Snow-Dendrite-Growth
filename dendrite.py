"""
    Here the phase field model of dendritic solidification refer to this thesis:
        Kobayashi, R. (1993), "Modeling and numerical simulations of dendritic crystal growth." 
        Physica D 63(3-4): 410-423
    
    a larger delta (anisotropic strength) can force the snow to grow on specific direction, 
    hence can give a more beautiful morphology
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

n = 512
phi = ti.field(dtype=ti.float64, shape=(n, n))
phiNew = ti.field(dtype=ti.float64, shape=(n, n))
tp = ti.field(dtype=ti.float64, shape=(n, n))  # temperature
tpNew = ti.field(dtype=ti.float64, shape=(n, n))
dEnergy_dGrad_term1 = ti.Vector.field(2, ti.float64, shape=(n, n))  # the firat term of the energy-derivative with respect to phi_grad
epsilons = ti.field(dtype=ti.float64, shape=(n, n))

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
angle0 = 0.  # np.pi / 18. * 1.5
showFrameFrequency = 16
writeImages = input("\033[35;1m write the output images as files? (y/n): \033[0m")


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
def divergence_dEnergy_dGrad_term1(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    return (dEnergy_dGrad_term1[ip, j][0] - dEnergy_dGrad_term1[im, j][0]) / (2. * dx) + \
           (dEnergy_dGrad_term1[i, jp][1] - dEnergy_dGrad_term1[i, jm][1]) / (2. * dx)


@ti.kernel
def get_epsilons_and_dEnergy_dGrad_term1():
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)
        grad = ti.Vector([
            (phi[ip, j] - phi[im, j]) / (2. * dx), 
            (phi[i, jp] - phi[i, jm]) / (2. * dx)
        ])
        gradNorm = sumVec(grad**2)
        if gradNorm < 1.e-8:
            dEnergy_dGrad_term1[i, j] = ti.Vector([0., 0.])
            angle = ti.atan2(grad[1], grad[0])
            epsilons[i, j] = epsilonbar * (1. + delta * ti.cos(anisoMod * (angle - angle0)))
        else:
            angle = ti.atan2(grad[1], grad[0])
            epsilon = epsilonbar * (1. + delta * ti.cos(anisoMod * (angle - angle0)))
            epsilons[i, j] = epsilon
            dAngle_dGradX = -grad[1] / gradNorm
            dAngle_dGradY = grad[0] / gradNorm
            tmp = epsilonbar * delta * -ti.sin(anisoMod * (angle - angle0)) * anisoMod
            dEpsilon_dGrad = tmp * ti.Vector([dAngle_dGradX, dAngle_dGradY])
            dEnergy_dGrad_term1[i, j] = epsilon * dEpsilon_dGrad * gradNorm


@ti.kernel
def evolution():
    """get phi and temperature at next step"""
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)

        lapla_phi = (  # laplacian of phi
            2 * (phi[im, j] + phi[i, jm] + phi[ip, j] + phi[i, jp]) 
            + (phi[im, jm] + phi[im, jp] + phi[ip, jm] + phi[ip, jp]) 
            - 12 * phi[i, j]
        ) / (3. * dx * dx)
        lapla_tp = (  # laplacian of temperature
            2 * (tp[im, j] + tp[i, jm] + tp[ip, j] + tp[i, jp]) 
            + (tp[im, jm] + tp[im, jp] + tp[ip, jm] + tp[ip, jp]) 
            - 12 * tp[i, j]
        ) / (3. * dx * dx)

        m_chem = alpha * ti.atan2(gamma * (teq - tp[i, j]), 1.)
        chemicalForce = phi[i, j] * (1. - phi[i, j]) * (phi[i, j] - 0.5 + m_chem)
        gradForce_term1 = divergence_dEnergy_dGrad_term1(i, j)
        grad_epsilon2 = ti.Vector([
            (epsilons[ip, j]**2 - epsilons[im, j]**2) / (2. * dx),
            (epsilons[i, jp]**2 - epsilons[i, jm]**2) / (2. * dx),
        ])
        grad_phi = ti.Vector([
            (phi[ip, j] - phi[im, j]) / (2. * dx), 
            (phi[i, jp] - phi[i, jm]) / (2. * dx)
        ])
        gradForce_term2 = grad_epsilon2[0] * grad_phi[0] + \
                grad_epsilon2[1] * grad_phi[1] + \
                epsilons[i, j]**2 * lapla_phi

        phiRate = mo * (chemicalForce + gradForce_term1 + gradForce_term2)
        phiNew[i, j] = phi[i, j] + phiRate * dt

        ### update the temperature
        tpRate = lapla_tp + k * phiRate
        tpNew[i, j] = tp[i, j] + tpRate * dt


@ti.kernel
def updateVariables():
    for i, j in phi:
        phi[i, j] = phiNew[i, j]
        tp[i, j] = tpNew[i, j]


def substeps():
    get_epsilons_and_dEnergy_dGrad_term1()
    evolution()
    updateVariables()


if __name__ == "__main__":
    
    initializeVariables()
    gui_tp = ti.GUI("temperature field", res=(n, n))
    gui_phi = ti.GUI("phase field", res=(n, n))

    for i in range(1000000):
        
        if i % showFrameFrequency == 0:
            gui_tp.set_image(tp)
            gui_tp.show()
            gui_phi.set_image(phi)
            gui_phi.show(
                "./pictures/{}.png".format(i) 
                    if i % (showFrameFrequency * 16) == 0 and writeImages == "y" 
                    else None
            )
    
        substeps()
