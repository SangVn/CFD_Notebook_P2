#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 

import numpy as np

#Burgers equation: du/dt + dF/dx = 0; F = u**2/2
epsilon = 1e-6

#Godunov scheme
def decay_godunov(ul, ur): #uL, uR là hai số
    ustar = 0
    if ul > ur:
        if (ul + ur)/2. > 0: ustar = ul
        else: ustar = ur
    else:
        if ul > 0: ustar = ul
        elif ur < 0: ustar = ur
        else: ustar = 0
    return ustar

#solver
def godunov(u, flux, dx, dt, nt):
    for n in range(nt):
        ustar = np.array([decay_godunov(ul, ur) for ul, ur in zip(u[:-1], u[1:])])
        u[1:-1] = u[1:-1] - dt/dx*(flux(ustar[1:]) - flux(ustar[:-1]))
    return

#MUSCL schemes
#slope limiter: phi(r), rs - mảng r
def minmod(rs):
    phi = np.array([max(0, min(1,r)) for r in rs])
    return phi

def koren(rs):
    phi = np.array([max(0, min(2*r, min((1+2*r)/3., 2))) for r in rs])
    return phi

def superbee(rs):
    phi = np.array([max(0, min(2*r, 1), min(r,2)) for r in rs])
    return phi

def vanleer74(rs):
    phi = np.array([(r+abs(r))/(1+abs(r)) for r in rs])
    return phi

def vanleer77(rs):
    phi = np.array([max(0, min(2*r, 0.5*(1+r),2)) for r in rs])
    return phi

def ospre(rs):
    phi = np.array([1.5*(r+r*r)/(1+r+r*r) for r in rs])
    return phi

# solver
def muscl(u, limiter, flux, dx, dt, nt):
    uL = u.copy()
    uR = u.copy()
    r  = u.copy()
    for n in range(nt):
        #tìm r, để tránh chia cho 0 ta cộng vào mẫu số 1e-6
        r[1:-1] = (u[1:-1] - u[:-2])/((u[2:] - u[1:-1]) + 1e-6)
        
        #tái cấu trúc nghiệm u_{i+1/2}^L
        uL[1:-2] = u[1:-2] + 0.5*limiter(r[1:-2])*(u[2:-1] - u[1:-2])
        
        #tái cấu trúc nghiệm u_{i+1/2}^R
        uR[1:-2] = u[2:-1] - 0.5*limiter(r[2:-1])*(u[3:] - u[2:-1])
        
        #giải bài toán riemann 
        ustar = np.array([decay_godunov(ul, ur) for ul, ur in zip(uL[1:-2], uR[1:-2])])

        #tính hàm dòng và tìm nghiệm u ở bước thời gian tiếp theo 
        u[2:-2] = u[2:-2] - dt/dx*(flux(ustar[1:]) - flux(ustar[:-1]))
        
        #điều kiện biên không đổi 
    return

#WENO schemes

# 1. WENO5 Interpolation
#các hệ số trong công thức chỉ thị độ trơn 
a_b0 = [4., -19., 25., 11., -31., 10.]
a_b1 = [4., -13., 13., 5., -13., 4.]
a_b2 = [10., -31., 25., 11., -19., 4.]
a_bItp  = [a_b0, a_b1, a_b2]

#tái cấu trúc u_{i+1/2}
#trọng lượng tuyến tính 
gamma_Itp_p05 = np.array([1./16, 5./8, 5./16])

#các hệ số trong công thức xấp xỉ bậc ba 
a_u0 = [3./8, -5./4, 15./8]
a_u1 = [-1./8, 3./4, 3./8]
a_u2 = [3./8, 3./4, -1./8]
a_uItp_p05  = [a_u0, a_u1, a_u2]

#tái cấu trúc u_{i-1/2}
gamma_Itp_m05 = np.array([5./16, 5./8, 1./16])

#các hệ số trong công thức xấp xỉ bậc ba 
a_u0 = [-1./8, 3./4, 3./8]
a_u1 = [3./8, 3./4, -1./8]
a_u2 = [15./8, -5./4, 3./8]
a_uItp_m05  = [a_u0, a_u1, a_u2]


# 2. WENO5 Reconstruction
a_b0 = [1., -4., 3.]
a_b1 = [1., 0., -1.]
a_b2 = [3., -4., 1.]
a_bRct = [a_b0, a_b1, a_b2]

#reconstruction u_{i+1/2}
gamma_Rct_p05 = np.array([1./10, 3./5, 3./10])

a_u0 = [1./3, -7./6, 11./6]
a_u1 = [-1./6, 5./6, 1./3]
a_u2 = [1./3, 5./6, -1./6]
a_uRct_p05  = [a_u0, a_u1, a_u2]

#reconstruction u_{i-1/2}
gamma_Rct_m05 = np.array([3./10, 3./5, 1./10])

a_u0 = [-1./6, 5./6, 1./3]
a_u1 = [1./3, 5./6, -1./6]
a_u2 = [11./6, -7./6, 1./3]
a_uRct_m05  = [a_u0, a_u1, a_u2]

#WENO5 Interpolation: smothness indicators
def beta_Itp(a, u):
    b = 1./3*(a[0]*u[0]**2 + a[1]*u[0]*u[1] + a[2]*u[1]**2 + a[3]*u[0]*u[2] + a[4]*u[1]*u[2] + a[5]*u[2]**2)
    return b

#WENO5 Reconstruction: smothness indicators
def beta_Rct(a, u):
    b = 13./12*(u[0] - 2*u[1] + u[2])**2 + 0.25*(a[0]*u[0] + a[1]*u[1] + a[2]*u[2])**2
    return b

#Hàm tìm trọng lượng phi tuyến 
def omega(gamma, beta):
    omgs = gamma/(epsilon + beta)**2
    omgs_sum = sum(omgs)
    omg = omgs/omgs_sum
    return omg

#Hàm tái cấu trúc nghiệm
def reconstruction(u, beta, a_b, gamma, a_u, u_05):
    for i in range(2, len(u)-2): #for each cell in cells
        #bước 1. tính chỉ thị độ trơn trong từng khuôn S_j
        b0 = beta(a_b[0], u[i-2:i+1])
        b1 = beta(a_b[1], u[i-1:i+2])
        b2 = beta(a_b[2], u[i  :i+3])
        
        #bước 2. tìm trọng lượng phi tuyến từng khuôn
        omg = omega(gamma, np.array([b0, b1, b2]))
        
        #bước 3. tìm xấp xỉ bậc 3
        u0 = sum(a_u[0] * u[i-2:i+1])
        u1 = sum(a_u[1] * u[i-1:i+2])
        u2 = sum(a_u[2] * u[i  :i+3])
        
        #bước 4. tìm xấp xỉ bậc 5
        u_05[i] = sum(omg *[u0, u1, u2])
        
        #Điều kiện biên: u_05[0], u_05[1], u_05[-2], u_05[-1]
    return

#solver
def weno5(u, beta, a_b, gamma_p05, gamma_m05, a_up05, a_um05, flux, dx, dt, nt):
    u_p05 = u.copy()
    u_m05 = u.copy()
    un   = u.copy()
    for n in range(nt):
        un = u.copy()
        for stage in range(3): #runge-kutta 3
            #Step 1: reconstruction u_{i+1/2}, u_{i-1/2}
            reconstruction(u, beta, a_b, gamma_p05, a_up05, u_p05)
            reconstruction(u, beta, a_b, gamma_m05, a_um05, u_m05)

            #Step 2: riemann solver
            ustar = np.array([decay_godunov(ul, ur) for ul, ur in zip(u_p05[1:-2], u_m05[2:-1])])

            #Step 3: time intergration
            if stage == 0: u[2:-2] = un[2:-2] - dt/dx*(flux(ustar[1:]) - flux(ustar[:-1]))
            elif stage == 1: u[2:-2] = 0.75*un[2:-2] + 0.25*u[2:-2] - 0.25*dt/dx*(flux(ustar[1:]) - flux(ustar[:-1]))
            else: u[2:-2] = 1./3*un[2:-2] + 2./3*u[2:-2] - 2./3*dt/dx*(flux(ustar[1:]) - flux(ustar[:-1]))
    return 

#schemes
def weno5_reconstruction(u, flux, dx, dt, nt):
    weno5(u, beta_Rct, a_bRct, gamma_Rct_p05, gamma_Rct_m05, a_uRct_p05, a_uRct_m05, flux, dx, dt, nt)
    return

def weno5_interpolation(u, flux, dx, dt, nt):
    weno5(u, beta_Itp, a_bItp, gamma_Itp_p05, gamma_Itp_m05, a_uItp_p05, a_uItp_m05, flux, dx, dt, nt)
    return
