# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:32:23 2015

@author: ta240184
"""
import numpy as np
import math

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
    B[:] = beta[1+S]; # beta indices start with 1, not 0
    i,j,k = ind2sub((NX, NY, NZ),find(B ~= B[[-1,range(1,Nx-1)],:,:]))
    isb = EjHeat.__mysub2ind__(EjHeat.__damod__(i-1,NX),j,k)
    isf = EjHeat.__mysub2ind__(i,j,k)
    i,j,k = ind2sub((NX, NY, NZ),find(B ~= B[:,[end 1:end-1],:]))
    jsb = EjHeat.__mysub2ind__(i,EjHeat.__damod__(j-1,NY),k)
    jsf = EjHeat.__mysub2ind__(i,j,k)
    i,j,k = ind2sub((NX, NY, NZ),find(B ~= B[:,:,[end 1:end-1]]))
    ksb = EjHeat.__mysub2ind__(i,j,EjHeat.__damod__(k-1,NZ))
    ksf = EjHeat.__mysub2ind__(i,j,k)
    li = len(isf)
    lj = len(jsf)
    lk = len(ksf)
    l = li+lj+lk
    iind = [1:li]
    jind = li+[1:lj]
    kind = lj+li+[1:lk]
    Bi = (B[isf]-B[isb])./(B[isf]+B[isb])
    Bj = (B[jsf]-B[jsb])./(B[jsf]+B[jsb])
    Bk = (B[ksf]-B[ksb])./(B[ksf]+B[ksb])
    F2 = -2*[double(d=='x')*Bi[:];double(d=='y')*Bj[:];double(d=='z')*Bk[:]]
    Psi = sparse([isb isf jsb jsf ksb ksf ],...
    [iind iind jind jind kind kind],...
    -0.5/H*np.ones((1,2*l),N,l)
    D = sparse([iind iind jind jind kind kind],...
    [isf isb jsf jsb ksf ksb ],...
    2/H*[Bi -Bi Bj -Bj Bk -Bk ],l,N)
    [MM,NM,LM] = ndgrid(2*(math.cos((2*pi/NX)*[0:NX-1])-1)/H^2,...
    2*(math.cos((2*pi/NY)*[0:NY-1])-1)/H^2,...
    2*(math.cos((2*pi/NZ)*[0:NZ-1])-1)/H^2)
    PDM = [MM+NM+LM] 
    del MM, LM, NM
    PDM[1,1,1] = 1
    G,e,I = EjHeat.__bicgstab__(EjHeat.__AM__,F2,errtol,maxit)
    U = EjHeat.__poisson__(-Psi*G) 
    del global PDM
    B_l,RR = EjHeat.__findBAndRelResidual__(N,d)
    
    return U,B_l,RR,I

#------------------------------------------------------
def __findBAndRelResidual(N,d)__:
    global NX, NY, NZ, H, U, G, B, F2, isf, isb, jsf, jsb, ksf, ksb, iind, jind, kind, Psi, D
    i = range(1:N)
    B = B[:]
    IF = EjHeat.__damod__(i+1,NX)+math.floor((i-1)/NX)*NX
    IB = EjHeat.__damod__(i-1,NX)+math.floor((i-1)/NX)*NX
    US = U[IF]+U[IB] # begin evaluation of Laplacian
    BS1 = B.*( (U[IF]-U[IB])/2/H + (d=='x')) 
    del IF, IB
    if ~isempty(isb): # iind is "right jump" for isb
        BS1[isb] = BS1[isb] - B[np.transpose(isb)].*G[iind]/4 # (23)
        BS1[isf] = BS1[isf] + B[np.transpose(isf)].*G[iind]/4 # (23)
    B_l[1] = np.sum(BS1)/N 
    del BS1 # (22)
    
    JF = EjHeat.__damod__(i+NX,NX*NY)+math.floor((i-1)/(NX*NY))*NX*NY
    JB = EjHeat.__damod__(i-NX,NX*NY)+math.floor((i-1)/(NX*NY))*NX*NY
    US = US+U[JF]+U[JB] # continue Laplacian
    BS2 = B.*( (U[JF]-U[JB])/2/H + (d=='y'))
    del JF, JB
    if ~isempty(jsb): # jind is "back jump" for jsb
        BS2[jsb] = BS2[jsb] - B[np.transpose(jsb)].*G[jind]/4
        BS2[jsf] = BS2[jsf] + B[np.transpose(jsf)].*G[jind]/4
    B_l[2] = np.sum(BS2)/N 
    del BS2
    
    KF = EjHeat.__damod__(i+NX*NY,NX*NY*NZ)
    KB = EjHeat.__damod__(i-NX*NY,NX*NY*NZ)
    US = US+U[KF]+U[KB] # continue Laplacian
    BS3 = B.*( (U[KF]-U[KB])/2/H + (d=='z')) 
    del KF, KB
    if ~isempty(ksb): # kind is "top jump" for ksb,
        BS3[ksb] = BS3[ksb] - B[np.transpose(ksb)].*G[kind]/4
        BS3[ksf] = BS3[ksf] + B[np.transpose(ksf)].*G[kind]/4
    B_l[3] = np.sum(BS3)/N
    del BS3
    
    F = -Psi*F2
    if norm(F) < eps:
        RR = 0;
    else:
        RR = norm(F-(US-6*U)/H^2+Psi*(D*U))/norm(F);
    
    
    return B_l,RR

    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __mysub2ind__(i,j,k):
    global NX, NY, NZ
    if ~isempty(i):
        l = np.transpose(sub2ind((NX,NY,NZ),i,j,k))
    else:
        l = []
    
    return l



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __damod__(m,n):
    d = m - math.floor((m-1)./n).*n
    return d



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __poisson__(f):
    global NX, NY, NZ, N, PDM
    FM = fftn(reshape(f,(NX,NY,NZ))
    UM = FM./PDM
    y = real(reshape(ifftn(UM),N,1))
    return y



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __AM__(x):
    global Psi, D
    y = x-D*(EjHeat.__poisson__(Psi*x));
    return y


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def __bicgstab__(atv,b, errtol, kmax):
#    % Bi-CGSTAB solver for linear systems
#    % C. T. Kelley, December 16, 1994
#    % adapted by Andreas L. Wiegmann, March 23, 2006
    n, error, x, errtol = len(b), [], np.zeros(n,1), errtol*norm(b)
    rho, r, SHOW = np.zeros((kmax+1,1)), b, False
    hatr0, total_iters, k, rho[0], alpha, omega = r, 0, 0, 1, 1, 1
    v, p, rho[1] = np.zeros((n,1)), np.zeros((n,1)), np.transpose(hatr0)*r
    zeta = norm(r)
    error.append(zeta)
    
    while ((zeta > errtol) and (k < kmax)):
        k = k+1
        if omega==0:
            raise Exception('Bi-CGSTAB breakdown, omega=0')
        beta = (rho[k]/rho[k-1])*(alpha/omega)
        p = r+beta*(p - omega*v)
        v = atv(p)
        tau = np.transpose(hatr0)*v
        if tau==0:
            raise Exception('Bi-CGSTAB breakdown, tau=0')
        alpha = rho[k]/tau
        s = r-alpha*v;
        t = atv(s)
        tau = np.transpose(t)*t
        if tau==0:
            raise Exception('Bi-CGSTAB breakdown, t=0')
        omega = np.transpose(t)*s/tau
        rho[k+1] = -omega*(np.transpose(hatr0)*t)
        x = x+alpha*p+omega*s
        r = s-omega*t
        zeta = norm(r)
        total_iters = k
        error.append(zeta)
        if SHOW: 
            print(['BICGSTAB: ',sprintf('%4d, ',k),...
                'err = ',sprintf('%.16e ',error[k]/error[0])]);
                
    return x, error, total_iters

