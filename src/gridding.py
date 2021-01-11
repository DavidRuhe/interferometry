import numpy as np
import math
from special.i0 import i0
from tqdm import trange


def get_beta(W, alpha):

    # see Beatty et al. (2005)
    beta = np.pi*np.sqrt((W*W/alpha/alpha)*(alpha - 0.5)*(alpha - 0.5) - 0.8)

    return beta


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

        xref = math.ceil((xval - 0.5*W*dx)/dx)*dx

        for xndx in range(W):

            xg = xref + xndx*dx

            kndx = indx*W + xndx
            
            gcf_val = gcf_kaiser(xg-xval, Dx, beta)

            vis2[kndx] = visval*gcf_val

            x2[kndx] = xg
            y2[kndx] = yval
            

def grid_2d(u, v, vis, du, Nu, umin, dv, Nv, vmin, alpha, W, hflag_u, hflag_v):
    
    nvis = len(u)
    W2 = W ** 2
    
    gv = np.zeros((Nu, Nv), dtype=complex)
    
    tv1 = np.zeros(W)
    tu1 = np.zeros(W)
    tvis1 = np.zeros(W, dtype=complex)
    
    tv2 = np.zeros(W2)
    tu2 = np.zeros(W2)
    tvis2 = np.zeros(W2, dtype=complex)
    
    ug = np.zeros(nvis * W2)
    vg = np.zeros(nvis * W2)
    visg = np.zeros(nvis * W2, dtype=complex)
    
    su = np.zeros(1)
    sv = np.zeros(1)
    svis = np.zeros(1, dtype=complex)
    
    beta = get_beta(W, alpha)

    
    from tqdm import trange
    
    for i in trange(nvis):
        
#         su = u[i]
#         sv = v[i]
        
        su[0] = u[i]
        sv[0] = v[i]
        svis[0] = vis[i]
        
        grid_1d_from_2d(su, svis, du, W, beta, sv, tu1, tvis1, tv1)
        
        grid_1d_from_2d(tv1, tvis1, dv, W, beta, tu1, tv2, tvis2, tu2)
        
        ug[i*W2:(i+1)*W2] = tu2
        vg[i*W2:(i+1)*W2] = tv2
        visg[i*W2:(i+1)*W2] = tvis2
        
        
    N = ug.shape[0]
    temp = 0.
    
    for i in trange(N):
        print(i)
        print('sumstart', gv.sum())
        temp = (ug[i] - umin)/du + 0.5
        undx = int(temp)
#         if i == 45:
        print(temp ,undx)
          
#         print(temp, undx)
        temp = (vg[i] - vmin)/dv + 0.5
        vndx = int(temp)
#         if i == 45:
        print(temp, vndx)
 
#         print(temp, vndx)
#         if i == 45: 
        print('gvvis')
        print(gv[undx, vndx])
        print(visg[i])

        if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
            print(True)
            gv[undx, vndx] = gv[undx, vndx] + visg[i]
            
        print('summid', gv.sum())
            
        print('gvmid')
        print(gv[undx, vndx])
            
        
#         print(gv[undx, vndx])
            
        if hflag_u or hflag_v:
            if hflag_u:
                temp = (-1.*ug[i] - umin)/du + 0.5
                undx = int(temp)
#                 print(temp, undx)
#                 if i == 45:
                print('hflag_u', temp, undx)

            if hflag_v:
                temp = (-1.*vg[i] - vmin)/dv + 0.5
                vndx = int(temp)
#                 print(temp, vndx)
#                 if i == 45:
                print('hflag_v', temp, vndx)
            
#             if i == 45
            print('gvis')
            print(gv[undx, vndx])
            print(visg[i].conjugate())
            
            if (undx>=0 and undx<Nu) and (vndx>=0 and vndx<Nv):
                print(True)
                
                print('here', gv[undx, vndx], visg[i].conjugate(), gv[undx, vndx] + visg[i].conjugate())
                print(gv.sum())
                gv[undx, vndx] = gv[undx, vndx] + visg[i].conjugate()
                print(gv.sum())
                print(gv[undx, vndx])
            
            print('gv')
            print(gv[undx, vndx])
            print('summend', gv.sum())
                
#             print(gv[undx, vndx])
        
        print(gv.sum())
#         print(gv.sum())
        
        if i == 45:
            raise
            
        if i == 75:
            raise
            
#         raise

    print(gv.sum())
    raise
                
    
    return gv


from scipy.constants import pi

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


def get_grid_corr_2d(dx, Nx, xmin, dy, Ny, ymin, du, dv, W, alpha):

        gridcorr = np.zeros([Ny, Nx])

        x = np.arange(Nx, dtype=float)*dx + xmin
        y = np.arange(Ny, dtype=float)*dy + ymin

        # see Beatty et al. (2005)
        beta = get_beta(W, alpha)

        for i in trange(Nx):
            for j in range(Ny):
                gridcorr[i,j] = inv_gcf_kaiser(x[i], du, W, beta)*\
                    inv_gcf_kaiser(y[j], dv, W, beta)

        return gridcorr



