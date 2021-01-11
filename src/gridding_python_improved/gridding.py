#cython: boundscheck=False
#cython: wraparound=False
"""
gridding.pyx

This file contains the gridding functions used in the GFFT package. We use
a gridding procedure as described in the paper

Beatty, P.J. and Nishimura, D.G. and Pauly, J.M. "Rapid gridding reconstruction
with a minimal oversampling ratio", IEEE Transactions on Medical Imaging, Vol. 24,
Num. 6, 2005
"""

"""
Copyright 2012 Michael Bell, Henrik Junklewitz

This file is part of GFFT.

GFFT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GFFT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GFFT.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
# cimport numpy as np
# cimport cython
# from cpython cimport bool
from special import i0, chbevl
from special.i0 import A, B
from scipy.constants import pi

DTYPE = np.float64
CTYPE = np.complex128

################################################################################
# 2D functions
################################################################################


def grid_1d_from_2dii(x, vis, dx, W, beta, y):
    nvis, N = x.shape
    Dx = W * dx
    visval = vis
    xval = x
    yval = y
    xref = np.ceil((xval - 0.5*W*dx)/dx)*dx
    xndx = np.arange(W)

    xg = xref + xndx * dx
    gcf_val = gcf_kaiseri(xg - xval, Dx, beta)
    
    vis2 = np.matmul(visval[:, :, None], gcf_val[:, None, :])
    vis2 = vis2.reshape(nvis, -1)
    x2 = np.tile(xg, N)
    y2 = np.repeat(yval, W, axis=-1)
    return x2, vis2, y2

def grid_2dii(u, v, vis, du, Nu, umin, dv, Nv, vmin, alpha, W, hflag_u, hflag_v):
    
    beta = np.array([get_beta(W, alpha)])
    
    tu1, tvis1, tv1 = grid_1d_from_2dii(u[:, None], vis[:, None], du, W, beta, v[:, None])
    tv2, tvis2, tu2 = grid_1d_from_2dii(tv1, tvis1, dv, W, beta, tu1) # output arrays
    
    ug = tu2.ravel()
    vg = tv2.ravel()
    visg = tvis2.ravel()
    
#     print(ug, vg, visg)
    

    gvi = np.zeros((Nu, Nv), dtype=complex) #output array

    undxi = ((ug - umin) / du + 0.5).astype(int)
    vndxi = ((vg - vmin) / du + 0.5).astype(int)

    mask = ((undxi >= 0) & (undxi < Nu)) & ((vndxi >= 0) & (vndxi < Nv))

    np.add.at(gvi, (undxi[mask], vndxi[mask]), visg[mask])# print(gv.sum())
    if hflag_u or hflag_v:
        if hflag_u:
            undxi = ((-1.*ug - umin)/du + 0.5).astype(int)
        if hflag_v:
            vndxi = ((-1.*vg - vmin)/dv + 0.5).astype(int)



        mask = ((undxi >= 0) & (undxi < Nu)) & ((vndxi >= 0) & (vndxi < Nv))    

        np.add.at(gvi, (undxi[mask], vndxi[mask]), visg.conjugate()[mask])
        
        
#     print(gvi.sum(), 'sum')
#     return ug, vg, visg, gvi
    return gvi
    
    

def grid_2d(u, v, vis, du, Nu, umin, dv, Nv, vmin, alpha, W, hflag_u, hflag_v):
    
        W2 = W**2
        nvis = u.shape[0]

        gv = np.zeros((Nu, Nv), dtype=CTYPE) #output array

        ug = np.zeros(nvis*W2, dtype=DTYPE)
        vg = np.zeros(nvis*W2, dtype=DTYPE)
        visg = np.zeros(nvis*W2, dtype=CTYPE)

        # holds the W values after u gridding
        tu1 = np.zeros(W, dtype=DTYPE)
        tvis1 = np.zeros(W, dtype=CTYPE)
        tv1 = np.zeros(W, dtype=DTYPE)


        # holds the W**2 values after subsequent v gridding
        tu2 = np.zeros(W2, dtype=DTYPE)
        tvis2 = np.zeros(W2, dtype=CTYPE)
        tv2 = np.zeros(W2, dtype=DTYPE)

        su = np.zeros(1, dtype=DTYPE)
        sv = np.zeros(1, dtype=DTYPE)
        svis = np.zeros(1, dtype=CTYPE)

        beta = get_beta(W, alpha)
        
        for i in range(nvis):

            # For each visibility point, grid in 3D, one dimension at a time
            # so each visibility becomes W**3 values located on the grid

            # Grid in u
            su[0] = u[i]
            sv[0] = v[i]
            svis[0] = vis[i]
            grid_1d_from_2d(su, svis, du, W, beta, sv, tu1, tvis1, tv1)
            
            # Grid in v
            grid_1d_from_2d(tv1, tvis1, dv, W, beta, tu1, \
                tv2, tvis2, tu2) # output arrays

            ug[i*W2:(i+1)*W2] = tu2
            vg[i*W2:(i+1)*W2] = tv2
            visg[i*W2:(i+1)*W2] = tvis2
            
        N = ug.shape[0]
        temp = 0

        for i in range(N):
            # compute the location for the visibility in the visibility cube
            temp = (ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (vg[i] - vmin)/dv + 0.5
            vndx = int(temp)

            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                gv[undx, vndx] = gv[undx, vndx] + visg[i]

            # now compute the location for the -u,-v,-l2 visibility, which is
            # equal to the complex conj of the u,v,l2 visibility if we
            # assume that the individual Stokes images in Faraday space are real

            if hflag_u or hflag_v:
                if hflag_u:
                    temp = (-1.*ug[i] - umin)/du + 0.5
                    undx = int(temp)
                if hflag_v:
                    temp = (-1.*vg[i] - vmin)/dv + 0.5
                    vndx = int(temp)

                if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                    gv[undx, vndx] = gv[undx, vndx] + visg[i].conjugate()

#         return ug, vg, visg, gv
        return gv

    

def grid_2di(u, v, vis, du, Nu, umin, dv, Nv, vmin, alpha, W, hflag_u, hflag_v):

        
        W2 = W**2
        nvis = u.shape[0]

        gv = np.zeros((Nu, Nv), dtype=CTYPE) #output array

        ug = np.zeros(nvis*W2, dtype=DTYPE)
        vg = np.zeros(nvis*W2, dtype=DTYPE)
        visg = np.zeros(nvis*W2, dtype=CTYPE)

        # holds the W values after u gridding
        tu1 = np.zeros(W, dtype=DTYPE)
        tvis1 = np.zeros(W, dtype=CTYPE)
        tv1 = np.zeros(W, dtype=DTYPE)


        # holds the W**2 values after subsequent v gridding
        tu2 = np.zeros(W2, dtype=DTYPE)
        tvis2 = np.zeros(W2, dtype=CTYPE)
        tv2 = np.zeros(W2, dtype=DTYPE)

        su = np.zeros(1, dtype=DTYPE)
        sv = np.zeros(1, dtype=DTYPE)
        svis = np.zeros(1, dtype=CTYPE)

        beta = get_beta(W, alpha)
        
        
#         print(u.shape, u.dtype)
#         print(vis.shape, vis.dtype)
#         print(du)
#         print(W)
#         print(beta)
#         print(v, v.dtype)
#         print(tu1.shape, vis.dtype)
#         print(tvis1.shape, tvis1.dtype)
#         print(tv1.shape, tv1.dtype)
        
        tu1, tvis1, tv1 = grid_1d_from_2di(u, vis, du, W, beta, v, tu1, tvis1, tv1)
        print(tv1.sum(), tvis1.sum(), tu1.sum())
        raise

        

        for i in range(nvis):

            # For each visibility point, grid in 3D, one dimension at a time
            # so each visibility becomes W**3 values located on the grid

            # Grid in u
            su[0] = u[i]
            sv[0] = v[i]
            svis[0] = vis[i]
            grid_1d_from_2d(su, svis, du, W, beta, sv, tu1, tvis1, tv1)

            # Grid in v
            grid_1d_from_2d(tv1, tvis1, dv, W, beta, tu1, \
                tv2, tvis2, tu2) # output arrays

            ug[i*W2:(i+1)*W2] = tu2
            vg[i*W2:(i+1)*W2] = tv2
            visg[i*W2:(i+1)*W2] = tvis2
            
        
        return ug, vg, visg


        N = ug.shape[0]
        temp = 0

        for i in range(N):
            # compute the location for the visibility in the visibility cube
            temp = (ug[i] - umin)/du + 0.5
            undx = int(temp)
            temp = (vg[i] - vmin)/dv + 0.5
            vndx = int(temp)

            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                gv[undx, vndx] = gv[undx, vndx] + visg[i]


            # now compute the location for the -u,-v,-l2 visibility, which is
            # equal to the complex conj of the u,v,l2 visibility if we
            # assume that the individual Stokes images in Faraday space are real

            if hflag_u or hflag_v:
                if hflag_u:
                    temp = (-1.*ug[i] - umin)/du + 0.5
                    undx = int(temp)
                if hflag_v:
                    temp = (-1.*vg[i] - vmin)/dv + 0.5
                    vndx = int(temp)

                if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                    gv[undx, vndx] = gv[undx, vndx] + visg[i].conjugate()

        return gv

       
    
# def degrid_2d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[DTYPE_t,ndim=1] v, \
#     np.ndarray[CTYPE_t, ndim=2] regVis, double du, int Nu, double umin, \
#     double dv, int Nv, double vmin, double alpha, int W):

#         cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
#             np.arange(0.,Nu,1.)*du + umin
#         cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] vgrid = \
#             np.arange(0.,Nv,1.)*dv + vmin

#         cdef int nvis = u.shape[0]

#         cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
#             np.zeros(nvis, dtype=CTYPE)

#         # From Beatty et al. (2005)
#         cdef double beta = get_beta(W, alpha)
#         # Grid in u and v
#         cdef double Du = W*du
#         cdef double Dv = W*dv

#         cdef Py_ssize_t i, j, urang, vrang, k, Wu, Wv

#         cdef double gcf_val, gcf_val_u, gcf_val_v


#         for k in range(nvis):

#             urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))
#             vrang = int(np.ceil((v[k] - 0.5*Dv - vmin)/dv))

#             for i in range(urang, urang+W):
#                 if (i>=Nu or i<0): continue
#                 gcf_val_u = gcf_kaiser(u[k]-ugrid[i], Du, beta)
#                 for j in range(vrang, vrang+W):
#                     if (j>=Nv or j<0): continue

#                     gcf_val_v = gcf_kaiser(v[k]-vgrid[j], Dv, beta)

#                     # convolution kernel for position i,j
#                     gcf_val = gcf_val_v*gcf_val_u
#                     #sampling back to visibility point k
#                     Vis[k] = Vis[k] + regVis[i,j]*gcf_val

#         return Vis


def get_grid_corr_2d(dx, Nx, xmin, dy, Ny, ymin, du, dv, W, alpha):

    gridcorr = np.zeros([Ny, Nx], dtype=DTYPE)

    x = np.arange(Nx, dtype=DTYPE)*dx + xmin
    y = np.arange(Ny, dtype=DTYPE)*dy + ymin

    # see Beatty et al. (2005)
    beta = get_beta(W, alpha)

    for i in range(Nx):
        for j in range(Ny):
            gridcorr[i,j] = inv_gcf_kaiser(x[i], du, W, beta)*\
                inv_gcf_kaiser(y[j], dv, W, beta)

    return gridcorr

    
def grid_1d_from_2d(x, vis, dx, W, beta, y, x2, vis2, y2):
    """
    Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
    """
    N = len(x)

    Dx = W*dx

#         cdef Py_ssize_t indx, xndx, kndx

#         cdef double xval, yval, xref, xg, gcf_val

#         cdef CTYPE_t visval

    for indx in range(N):

        visval = vis[indx]

        xval = x[indx]
        yval = y[indx]
        
        xref = np.ceil((xval - 0.5*W*dx)/dx)*dx

        for xndx in range(W):
            
            xg = xref + xndx*dx

            kndx = indx*W + xndx
            
            gcf_val = gcf_kaiser(xg-xval, Dx, beta)
            
            vis2[kndx] = visval*gcf_val

            x2[kndx] = xg
            y2[kndx] = yval
            
            
def grid_1d_from_2di(x, vis, dx, W, beta, y, x2, vis2, y2):
    """
    Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
    """
    Dx = W*dx

    visval = vis.copy()
    xval = x.copy()
    yval = y.copy()
    xref = np.ceil((xval - 0.5*W*dx)/dx)*dx
    
    xg = xref[:, None] + np.arange(W) * dx
    
    gcf_val = gcf_kaiser_improved(xg - xval[:, None], Dx, beta)
    vis2 = visval[:, None] * gcf_val
    x2 = xg.astype(complex)
    y2 = np.tile(y[:, None], (1, W))
    
    return x2, y2, vis2

#     for xndx in range(W):

#         xg = xref + xndx*dx

#         print(xndx)
#         kndx = xndx

#         gcf_val = gcf_kaiser(xg-xval, Dx, beta)

#         vis2[kndx] = visval*gcf_val

#         x2[kndx] = xg
#         y2[kndx] = yval

            
#     print('hi')
#     return x2, y2, vis2


# cdef inline void grid_1d_from_2d(np.ndarray[DTYPE_t,ndim=1] x, \
#     np.ndarray[CTYPE_t,ndim=1] vis, double dx, int W, double beta, \
#     np.ndarray[DTYPE_t,ndim=1] y, \
#     np.ndarray[DTYPE_t,ndim=1] x2, np.ndarray[CTYPE_t,ndim=1] vis2,\
#     np.ndarray[DTYPE_t,ndim=1] y2):


#         """
#         Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
#         """
#         cdef int N = x.shape[0]

#         cdef double Dx = W*dx

#         cdef Py_ssize_t indx, xndx, kndx

#         cdef double xval, yval, xref, xg, gcf_val

#         cdef CTYPE_t visval

#         for indx in range(N):

#             visval = vis[indx]

#             xval = x[indx]
#             yval = y[indx]

#             xref = ceil((xval - 0.5*W*dx)/dx)*dx

#             for xndx in range(W):

#                 xg = xref + xndx*dx

#                 kndx = indx*W + xndx

#                 gcf_val = gcf_kaiser(xg-xval, Dx, beta)

#                 vis2[kndx] = visval*gcf_val

#                 x2[kndx] = xg
#                 y2[kndx] = yval



################################################################################
# 1D functions
################################################################################

# def grid_1d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[CTYPE_t,ndim=1] vis, \
#     double du, int Nu, double umin, double alpha, int W, bool hermitianize):
#         """
#         Grid the data in w, Qvix, Uvis in 1D (x) and duplicate orthogonal axes
#         """
#         cdef int N = u.shape[0]
#         cdef double Du = W*du

#         cdef Py_ssize_t indx, undx, kndx
#         cdef double uval, uref, tu, gcf_val
#         cdef CTYPE_t visval
#         cdef double temp = 0.

#         cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ug = \
#             np.zeros(N*W, dtype=DTYPE)
#         cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] visg = \
#             np.zeros(N*W, dtype=CTYPE)

#         cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] gv = \
#             np.zeros(Nu, dtype=CTYPE) # output array

#         # From Beatty et al. (2005)
#         cdef double beta = get_beta(W, alpha)

#         # do convolution
#         for indx in range(N):

#             visval = vis[indx]
#             uval = u[indx]

#             uref = ceil((uval - 0.5*W*du - umin)/du)*du + umin

#             for undx in range(W):

#                 tu = uref + undx*du
#                 kndx = indx*W + undx

#                 gcf_val = gcf_kaiser(tu-uval, Du, beta)

#                 visg[kndx] = visval*gcf_val
#                 ug[kndx] = tu

#         # sample onto grid
#         for indx in range(N*W):
#             # compute the location for the visibility in the visibility cube
#             temp = (ug[indx] - umin)/du + 0.5
#             undx = int(temp)

#             if (undx>=0 and undx<Nu):
#                     gv[undx] = gv[undx] + visg[indx]

#             if hermitianize:
#                 temp = (-1.*ug[indx] - umin)/du + 0.5
#                 undx = int(temp)

#                 if (undx>=0 and undx<Nu):
#                     gv[undx] = gv[undx] + visg[indx].conjugate()

#         return gv

# def degrid_1d(np.ndarray[DTYPE_t,ndim=1] u, np.ndarray[CTYPE_t, ndim=1] regVis,\
#     double du, int Nu, double umin, double alpha, int W):

#         cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ugrid = \
#             np.arange(0.,Nu,1.)*du + umin

#         cdef int nvis = u.shape[0]

#         cdef np.ndarray[CTYPE_t, ndim=1, mode='c'] Vis = \
#             np.zeros(nvis, dtype=CTYPE)

#         # From Beatty et al. (2005)
#         cdef double beta = get_beta(W, alpha)
#         # Grid in u and v
#         cdef double Du = W*du

#         cdef Py_ssize_t i, j, urang, k

#         cdef double gcf_val

#         for k in range(nvis):

#             urang = int(np.ceil((u[k] - 0.5*Du - umin)/du))

#             for i in range(urang, urang+W):
#                 if (i<Nu and i>=0):
#                     #convolution kernel for position i
#                     gcf_val = gcf_kaiser(u[k]-ugrid[i], Du, beta)
#                     #sampling back to visibility point k
#                     Vis[k] = Vis[k] + regVis[i]*gcf_val

#         return Vis


# def get_grid_corr_1d(double dx, int Nx, double xmin, double du, int W, \
#     double alpha):

#         cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] gridcorr = np.zeros(Nx,\
#             dtype=DTYPE)
#         cdef np.ndarray[DTYPE_t,ndim=1, mode='c'] x = np.arange(Nx,\
#             dtype=DTYPE)*dx + xmin

#         cdef double beta = get_beta(W, alpha)

#         cdef Py_ssize_t i

#         for i in range(Nx):
#             gridcorr[i] = inv_gcf_kaiser(x[i], du, W, beta)

#         return gridcorr


# def test_gcf_kaiser(double k, double dk, int W, double alpha):

#     cdef double beta = get_beta(W, alpha)
#     return gcf_kaiser(k, dk*W, beta)

################################################################################
# Common functions
################################################################################

def get_beta(W, alpha):

    # see Beatty et al. (2005)
    beta = np.pi*np.sqrt((W*W/alpha/alpha)*(alpha - 0.5)*(alpha - 0.5) - 0.8)

    return beta



# cdef inline double get_beta(int W, double alpha):
#     cdef double pi = 3.141592653589793
#     # see Beatty et al. (2005)
#     cdef double beta = pi*sqrt((W*W/alpha/alpha)*(alpha - 0.5)*(alpha - 0.5) \
#         - 0.8)

#     return beta

# cdef inline double gcf_kaiser(double k, double Dk, double beta):

#     cdef double temp3 = 2.*k/Dk

#     if (1 - temp3)*(1 + temp3) < -1e-12:
# #        print "There is an issue with the gridding code!"
#         raise Exception("There is an issue with the gridding code!")

#     temp3 = sqrt(abs((1 - temp3)*(1 + temp3)))

#     temp3 = beta*temp3

# #    cdef double C = (1./Dk)*gsl_sf_bessel_I0(temp3)/gsl_sf_bessel_I0(beta)
#     cdef double C = gsl_sf_bessel_I0(temp3)/gsl_sf_bessel_I0(beta)

#     return C


# cdef inline double inv_gcf_kaiser(double x, double dk, int W, double beta):

#     cdef double pi = 3.141592653589793
#     cdef double temp1 = pi*pi*W*W*dk*dk*x*x
#     cdef double temp2 = beta*beta
#     cdef double temp, c

#     temp = sqrt(temp2 - temp1)
#     c0 = (exp(beta)-exp(-1.*beta))/2./beta
#     c = (exp(temp) - exp(-1.*temp))/2./temp

#     if temp1>temp2:
#         temp = sqrt(temp1 - temp2)
#         c = -0.5*(exp(-1.*temp) - exp(temp))/temp
#         #print "WARNING: There may be trouble..."


#     return c/c0


def inv_gcf_kaiseri(x, dk, W, beta):
    
    temp1 = np.pi*np.pi*W*W*dk*dk*x*x
    temp2 = beta*beta

    temp = np.sqrt(temp2 - temp1)
    c0 = (np.exp(beta)-np.exp(-1.*beta))/2./beta
    c = (np.exp(temp) - np.exp(-1.*temp))/2./temp
    
    tdg = temp1 > temp2
    
    if tdg.any():
        temp = temp1[tdg] - temp2
        c[tdg] = -0.5*(np.exp(-1.*temp) - np.exp(temp))/temp

#         c[tdg] = -0.5*(np.exp(-1.*np.sqrt(temp))) - np.exp(np.sqrt(temp)))/np.sqrt(temp1 - temp2)

    return c/c0


def get_grid_corr_2di(dx, Nx, xmin, dy, Ny, ymin, du, dv, W, alpha):

    x = np.arange(Nx)*dx + xmin
    y = np.arange(Ny)*dy + ymin

    # see Beatty et al. (2005)
    beta = get_beta(W, alpha)
    
    x_inv_kaiser = inv_gcf_kaiseri(x, du, W, beta)
    y_inv_kaiser = inv_gcf_kaiseri(y, dv, W, beta)
    
    gridcorr = np.outer(x_inv_kaiser, y_inv_kaiser)
    return gridcorr



def inv_gcf_kaiser(x, dk, W, beta):
    
    temp1 = pi*pi*W*W*dk*dk*x*x
    temp2 = beta*beta

    temp = np.sqrt(temp2 - temp1)
    c0 = (np.exp(beta)-np.exp(-1.*beta))/2./beta
    c = (np.exp(temp) - np.exp(-1.*temp))/2./temp

    if temp1>temp2:
        temp = np.sqrt(temp1 - temp2)
        c = -0.5*(np.exp(-1.*temp) - np.exp(temp))/temp
        #print "WARNING: There may be trouble..."


    return c/c0


def gcf_kaiser(k, Dk, beta):

    temp3 = 2.*k/Dk
    if (1 - temp3)*(1 + temp3) < -1e-12:
#        print "There is an issue with the gridding code!"
        raise Exception("There is an issue with the gridding code!")

    temp3 = np.sqrt(abs((1 - temp3)*(1 + temp3)))

    temp3 = beta*temp3
    
#    cdef double C = (1./Dk)*gsl_sf_bessel_I0(temp3)/gsl_sf_bessel_I0(beta)
    C = i0(temp3) / i0(beta)

    return C


def i0ei(x):
    
    x[x < 0] *= 1
    
    y = np.zeros(x.shape)
    
    se8 = x <= 8.0
    
    y[se8] = chbevl(x[se8] / 2 - 2, A, 30)
    xg8 = x[~se8]
    y[~se8] = chbevl(32 / xg8 - 2, B, 25) / np.sqrt(xg8)
    return y
   
#     return (chbevl(32.0 / x - 2.0, B, 25) / np.sqrt(x))


def i0i(x):
    return np.exp(x) * i0ei(x)


def gcf_kaiseri(k, Dk, beta):
    temp3 = 2.*k/Dk

    if np.any((1 - temp3)*(1 + temp3) < -1e-12):
        raise Exception("There is an issue with the gridding code!")

    temp3 = np.sqrt(abs((1 - temp3)*(1 + temp3)))

    temp3 = beta*temp3
    
    # TODO: beta should be np array to also use i0i.
    C = i0i(temp3) / i0i(beta)

    return C

