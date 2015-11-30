# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:32:23 2015

@author: ta240184
"""
import numpy as np
import scipy.sparse as sp

def EJ_heat(S,beta,d,errtol,maxit):
#    % Explicit jump effective thermal conductivity
#    % Andreas L. Wiegmann, 25 March 2006.
#    % Input :
#    % S: 3d voxelindices (0-based)
#    % beta: conductivities, defined for all indices in S; add 1!
#    %       beta(1)=conductivity of phase 0, beta(n)=conductivity of phase n-1
#    % d: 'x','y' or 'z' direction,
#    % errtol: desired relative residual for Schur-complement, bicgstab
#    % maxit: iteration cap for BiCGStab
#    %
#    % Output :
#    % U     temperature field
#    % B_l   effective conductivity
#    % RR    relative residual
#    % I number of iterations
    
    global Psi, D, NX, NY, NZ, H, PDM, U, G, B, F2, isb, isf, jsb, jsf, ksb, ksf, iind, jind, kind
    NX,NY,NZ = np.shape(S)
    N = NX*NY*NZ
    H = (d=='x')/NX +(d=='y')/NY+(d=='z')/NZ
    B = np.zeros(np.shape(S)) # B same size as S also in 1d cases
#???    B[:] = beta[1+S] # beta indices start with 1, not 0
    
    shift=np.hstack((np.asarray([-1]),np.arange(0,Nx-1)))
    i,j,k = np.unravel_index(np.flatnonzero(B!=B[shift,:,:]),(NX, NY, NZ))
    isb = __mysub2ind__(__damod__(i-1,NX),j,k)
    isf = __mysub2ind__(i,j,k)
    
    shift=np.hstack((np.asarray([-1]),np.arange(0,Ny-1)))
    i,j,k = np.unravel_index(np.flatnonzero(B != B[:,shift,:]),(NX, NY, NZ))
    jsb = __mysub2ind__(i,__damod__(j-1,NY),k)
    jsf = __mysub2ind__(i,j,k)
    
    shift=np.hstack((np.asarray([-1]),np.arange(0,Nz-1)))
    i,j,k = np.unravel_index(np.flatnonzero(B != B[:,:,shift]),(NX, NY, NZ))
    ksb = __mysub2ind__(i,j,__damod__(k-1,NZ))
    ksf = __mysub2ind__(i,j,k)
    
    li = len(isf)
    lj = len(jsf)
    lk = len(ksf)
    l = li+lj+lk
    iind = np.arange(1,li+1)
    jind = li+np.arange(1,lj+1)
    kind = lj+li+np.arange(1,lk+1)
    
    Bi = (B[isf]-B[isb])/(B[isf]+B[isb])
    Bj = (B[jsf]-B[jsb])/(B[jsf]+B[jsb])
    Bk = (B[ksf]-B[ksb])/(B[ksf]+B[ksb])
    
    F2 = -2*np.vstack((float(d=='x')*Bi[:],
                       float(d=='y')*Bj[:],
                       float(d=='z')*Bk[:]))
        
    row_ind = np.hstack((isb, isf, jsb, jsf, ksb, ksf))
    col_ind = np.hstack((iind, iind, jind, jind, kind, kind))
    data = -0.5/H*np.ones((1,2*l))
    Psi = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(N,l))
    
    row_ind = np.hstack((iind, iind, jind, jind, kind, kind))
    col_ind = np.hstack((isf, isb, jsf, jsb, ksf, ksb))
    data = 2/H*np.hstack((Bi, -Bi, Bj, -Bj, Bk, -Bk))
    D = sp.sparse.csr_matrix((data,(row_ind,col_ind)),shape=(l,N))
    
    MM,NM,LM = np.meshgrid(2*(np.cos((2*pi/NX)*np.arange(0,NX))-1)/H^2,
                           2*(np.cos((2*pi/NY)*np.arange(0,NY))-1)/H^2,
                           2*(np.cos((2*pi/NZ)*np.arange(0,NZ))-1)/H^2,
                           indexing='ij')
    PDM = MM+NM+LM 
    del MM, LM, NM
    PDM[0,0,0] = 1
    
    G,e,I = __bicgstab__(__AM__,F2,errtol,maxit)
    
    U = __poisson__(-Psi.dot(G)) 
    
    del PDM
    
    B_l,RR = __findBAndRelResidual__(N,d)
    
    return U,B_l,RR,I


#------------------------------------------------------
def __findBAndRelResidual__(N,d):
    global NX, NY, NZ, H, U, G, B, F2, isf, isb, jsf, jsb, ksf, ksb, iind, jind, kind, Psi, D
    i = np.arange(1,N+1)
    B = B[:]
    B_l = np.zeros(3)
    IF = __damod__(i+1,NX)+np.floor((i-1)/NX)*NX
    IB = __damod__(i-1,NX)+np.floor((i-1)/NX)*NX
    US = U[IF]+U[IB] # begin evaluation of Laplacian
    BS1 = B*( (U[IF]-U[IB])/2/H + (d=='x')) 
    del IF, IB
    if ~isempty(isb): # iind is "right jump" for isb
        BS1[isb] = BS1[isb] - B[np.transpose(isb)]*G[iind]/4 # (23)
        BS1[isf] = BS1[isf] + B[np.transpose(isf)]*G[iind]/4 # (23)
    B_l[0] = np.sum(BS1)/N 
    del BS1 # (22)
    
    JF = __damod__(i+NX,NX*NY)+np.floor((i-1)/(NX*NY))*NX*NY
    JB = __damod__(i-NX,NX*NY)+np.floor((i-1)/(NX*NY))*NX*NY
    US = US+U[JF]+U[JB] # continue Laplacian
    BS2 = B*( (U[JF]-U[JB])/2/H + (d=='y'))
    del JF, JB
    if ~isempty(jsb): # jind is "back jump" for jsb
        BS2[jsb] = BS2[jsb] - B[np.transpose(jsb)]*G[jind]/4
        BS2[jsf] = BS2[jsf] + B[np.transpose(jsf)]*G[jind]/4
    B_l[1] = np.sum(BS2)/N 
    del BS2
    
    KF = __damod__(i+NX*NY,NX*NY*NZ)
    KB = __damod__(i-NX*NY,NX*NY*NZ)
    US = US+U[KF]+U[KB] # continue Laplacian
    BS3 = B*( (U[KF]-U[KB])/2/H + (d=='z')) 
    del KF, KB
    if ~isempty(ksb): # kind is "top jump" for ksb,
        BS3[ksb] = BS3[ksb] - B[np.transpose(ksb)]*G[kind]/4
        BS3[ksf] = BS3[ksf] + B[np.transpose(ksf)]*G[kind]/4
    B_l[2] = np.sum(BS3)/N
    del BS3
    
    F = -Psi.dot(F2)
    if np.linalg.norm(F) < eps:
        RR = 0;
    else:
        RR = np.linalg.norm(F-(US-6*U)/H^2+Psi.dot(D.dot(U))
                                )/np.linalg.norm(F);
    
    
    return B_l,RR

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __mysub2ind__(i,j,k):
    global NX, NY, NZ
    if not len(i)==0:
        l = np.transpose(np.ravel_multi_index((i,j,k),(NX,NY,NZ)))
    else:
        l = np.array([])
    
    return l



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __damod__(m,n):
    d = m - np.floor((m-1)/n)*n
    return d



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __poisson__(f):
    global NX, NY, NZ, N, PDM
    FM = np.fft.fftn(np.reshape(f,(NX,NY,NZ)))
    UM = FM/PDM
    y = np.real(np.reshape(np.fft.ifftn(UM),(N,1)))
    return y



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __AM__(x):
    global Psi, D
    y = x-D.dot(__poisson__(Psi.dot(x)))
    return y


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __bicgstab__(atv,b, errtol, kmax):
#    % Bi-CGSTAB solver for linear systems
#    % C. T. Kelley, December 16, 1994
#    % adapted by Andreas L. Wiegmann, March 23, 2006
    n = len(b)
    error, x, errtol = [], np.zeros(n,1), errtol*np.linalg.norm(b)
    rho, r, SHOW = np.zeros((kmax+1,1)), b, False
    hatr0, total_iters, k, alpha, omega = r, 0, 0, 1, 1
    v, p = np.zeros((n,1)), np.zeros((n,1))
    rho[0] = 1
    rho[1] = np.matmul(np.transpose(hatr0),r)
    zeta = np.linalg.norm(r)
    error.append(zeta)
    
    while ((zeta > errtol) and (k < kmax)):
        k = k+1
        if omega==0:
            raise Exception('Bi-CGSTAB breakdown, omega=0')
        beta = (rho[k]/rho[k-1])*(alpha/omega)
        p = r+beta*(p - omega*v)
        v = atv(p)
        tau = np.matmul(np.transpose(hatr0),v)
        if tau==0:
            raise Exception('Bi-CGSTAB breakdown, tau=0')
        alpha = rho[k]/tau
        s = r-alpha*v;
        t = atv(s)
        tau = np.matmul(np.transpose(t),t)
        if tau==0:
            raise Exception('Bi-CGSTAB breakdown, t=0')
        omega = np.matmul(np.transpose(t),s)/tau
        rho[k+1] = -omega*(np.matmul(np.transpose(hatr0),t))
        x = x+alpha*p+omega*s
        r = s-omega*t
        zeta = np.linalg.norm(r)
        total_iters = k
        error.append(zeta)
        if SHOW: 
            print('BICGSTAB: %4d, err = %.16e ' %(k,error[k]/error[0]))
                
    return x, error, total_iters

