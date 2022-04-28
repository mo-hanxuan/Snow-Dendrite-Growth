import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n = 320
phi = ti.field(dtype=float, shape=(n, n))
phiNew = ti.field(dtype=float, shape=(n, n))

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
    radius = 32
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


@ti.func
def laplacian(i, j):
    im, jm, ip, jp = neighbor_index(i, j)
    return (phi[ip, j] + phi[im, j] - 2 * phi[i, j]) / dx ** 2 + \
           (phi[i, jp] + phi[i, jm] - 2 * phi[i, j]) / dy ** 2


@ti.kernel
def evolution():
    for i, j in phi:
        chemicalForce = u * (1. - 2. * phi[i, j])
        gradientForce = -k * laplacian(i, j)
        force = chemicalForce + gradientForce
        phi1 = phi[i, j] - mo * force * dt
        if phi1 > 1:
            phi1 = 1
        elif phi1 < 0:
            phi1 = 0
        phiNew[i, j] = phi1


@ti.kernel
def copyPhi():
    for i, j in phi:
        phi[i, j] = phiNew[i, j]


if __name__ == "__main__":

    initializePhi()
    gui = ti.GUI("phase field", res=(n, n), color=0x336699)
    
    for i in range(1000000):
        evolution()
        copyPhi()
        gui.set_image(phi)
        gui.show()