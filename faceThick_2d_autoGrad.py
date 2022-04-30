"""
    use autoGrad to compute derivative of energy
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n = 512
phi = ti.field(dtype=float, shape=(n, n), needs_grad=True)
phiGrad = ti.Vector.field(2, ti.float64, (n, n), needs_grad=True)
phiNew = ti.field(dtype=float, shape=(n, n))
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy

k = 64.  # gradient energy coefficient
u = 8. # chemical energy coefficient
mo = 0.02  # mobility

dt = 0.04
dx, dy = 1., 1.


@ti.func
def sumVec(vec):
    res = 0.
    for i in ti.static(range(vec.n)):
        res += vec[i]
    return res


@ti.kernel
def initializePhi():
    center = ti.Vector([n//2, n//2])
    radius = 64
    for i, j in phi:
        if sumVec((ti.Vector([i, j]) - center) ** 2) < radius ** 2:
            phi[i, j] = 1.
        else:
            phi[i, j] = 0.


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


@ti.kernel
def compute_phiGrad():
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)
        phiGrad[i, j][0] = (phi[ip, j] - phi[im, j]) / (2. * dx)
        phiGrad[i, j][1] = (phi[i, jp] - phi[i, jm]) / (2. * dy)


@ti.kernel
def compute_U():  # compute the total energy
    for i, j in phi:
        U[None] += u * phi[i, j] * (1. - phi[i, j]) + \
                   k * sumVec(phiGrad[i, j]**2)


@ti.kernel
def evolution():
    for i, j in phi:
        im, jm, ip, jp = neighbor_index(i, j)
        chemmicalForce = -phi.grad[i, j]
        gradientForce = (phiGrad.grad[ip, j][0] - phiGrad.grad[im, j][0]) / (2. * dx) + \
                        (phiGrad.grad[i, jp][1] - phiGrad.grad[i, jm][1]) / (2. * dy)
        phi1 = phi[i, j] + mo * (chemmicalForce + gradientForce) * dt
        if phi1 > 1:
            phi1 = 1
        elif phi1 < 0:
            phi1 = 0
        phiNew[i, j] = phi1


@ti.kernel
def copyPhi():
    for i, j in phi:
        phi[i, j] = phiNew[i, j]


def substeps():
    compute_phiGrad()
    with ti.Tape(loss=U):
        compute_U(
        )  # The tape will automatically compute dU/dx and save the results in x.grad
    evolution()
    copyPhi()


if __name__ == "__main__":

    initializePhi()
    gui = ti.GUI("phase field", res=(n, n))
    
    for i in range(1000000):
        substeps()

        if i % 16 == 0:
            gui.set_image(phi)
            gui.show()