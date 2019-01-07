#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 

# Giải hệ phương trình Euler 1D sơ đồ Godunov

import numpy as np
from scipy.optimize import fsolve, newton
from decay import decay_godunov
from constants import*
from convert_variables import*


#phương pháp lặp tìm P
def pressure_classic_godunov(Pl, Pr):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    #vận tốc âm thanh 
    c1 = np.sqrt(g*p1/r1)
    c2 = np.sqrt(g*p2/r2)
    
    #phương pháp lặp 
    P0 = (p1*r2*c2 + p2*r1*c1 + (u1-u2)*r1*c1*r2*c2)/(r1*c1+r2*c2)
    if P0 < eps: P0 = eps
    iterations = 50 # max_iteration 
    while (True):
        P = P0 #áp suất P^{n-1}
        if P >= p1: a1 = np.sqrt(r1*(gp1d2*P + gm1d2*p1))
        else:
            pp = max(eps, P/p1)
            op = 1. - pp**gm1d2g
            if op>=eps: a1 = gm1d2g*r1*c1*(1. - pp)/op
            else: a1 = r1*c1
        if P >= p2: a2 = np.sqrt(r2*(gp1d2*P + gm1d2*p2))
        else:
            pp = max(eps, P/p2)
            op = 1. - pp**gm1d2g
            if op>=eps: a2 = gm1d2g*r2*c2*(1. - pp)/op
            else: a2 = r2*c2

        z = P/(p1+p2)
        alpha = gm1/(3*g)*(1. - z)/np.power(z, gp1d2g)/(1. - np.power(z, gm1d2g)) - 1.
        if alpha < 0.: alpha = 0.
        phi = (a2*p1 + a1*p2 + a1*a2*(u1 - u2))/(a1+a2)

        P0 = (alpha*P + phi)/(1. + alpha)#tính P^n
        iterations -= 1
        if (abs(P0 - P) < eps) or (not iterations): break
    #kết thúc vòng lặp! 
  
    return P, c1, c2, a1, a2

#giải phương trình f(P) = 0 , tìm P
def pressure_root_finding(Pl, Pr):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    #vận tốc âm thanh 
    c1 = np.sqrt(g*p1/r1)
    c2 = np.sqrt(g*p2/r2)

    fl_shock = lambda P: (P - p1)/(r1*c1*np.sqrt(gp1d2g*(P/p1) + gm1d2g))
    fl_expansion = lambda P: c1/gm1d2*((P/p1)**gm1d2g - 1.)
    fr_shock = lambda P: (P - p2)/(r2*c2*np.sqrt(gp1d2g*(P/p2) + gm1d2g))
    fr_expansion = lambda P: c2/gm1d2*((P/p2)**gm1d2g - 1.)
    
    def fl(P):
        if P >= p1: return fl_shock(P)
        else: return fl_expansion(P)
    def fr(P):
        if P >= p2: return fr_shock(P)
        else: return fr_expansion(P)
    
    f = lambda P: fl(P) + fr(P) + u2 - u1
    
    P,info, ier, msg = fsolve(f, (p1+p2)/2.,full_output=True,xtol=1.e-14)
    #trường hợp sóng giãn mạnh hay chân không 
    if ier!=1:
        P,info, ier, msg = fsolve(f, (p1+p2)/2.,full_output=True,factor=0.1,xtol=1.e-10)
        # trường hợp nghiệm không hội tụ 
        if ier!=1: 
            print 'Warning: fsolve did not converge.'
            print msg
   
    if P >= p1-eps: a1 = np.sqrt(r1*(gp1d2*P + gm1d2*p1))
    else: a1 =  gm1d2g*r1*c1*(1. - P/p1)/(1. - np.power(P/p1, gm1d2g))
    if P >= p2-eps: a2 = np.sqrt(r2*(gp1d2*P + gm1d2*p2))
    else: a2 =  gm1d2g*r2*c2*(1. - P/p2)/(1. - np.power(P/p2, gm1d2g))
    
    return P, c1, c2, a1, a2

#tìm nghiệm chính xác bài toán Riemann 
def riemann_exact_solution(Pl, Pr, pressure_solver, x, xstar, time_target):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    P, c1, c2, a1, a2 = pressure_solver(Pl, Pr)
    U = (a1*u1 + a2*u2 + p1 - p2)/(a1 + a2)
    #xét gián đoạn bên trái 
    if P > p1: #nếu là sóng xung kích 
        D1 = u1 - a1/r1
        D1star = D1
        R1 = r1*a1/(a1-r1*(u1-U))
    else: #nếu là sóng giãn
        D1 = u1 - c1
        c1star = c1 + gm1d2*(u1-U)
        D1star = U - c1star
        R1 = g*P/c1star**2
    #tương tự cho gián đoạn bên phải 
    if P > p2: #nếu là sóng xung kích 
        D2 = u2 + a2/r2
        D2star = D2
        R2 = r2*a2/(a2 + r2*(u2-U))
    else: #nếu là sóng giãn
        D2 = u2 + c2
        c2star = c2 - gm1d2*(u2-U)
        D2star = U + c2star
        R2 = g*P/c2star**2
        
    #Căn cứ vào vị trí của các gián đoạn tại thời điểm t và vị trí điểm x_i 
    w = (x - xstar)/time_target
    P_out = np.zeros((3, len(x)))
    for i, wi in enumerate(w):
        if wi<=D1:   #nằm bên trái sóng trái
            P_out[0, i] = r1
            P_out[1, i] = u1
            P_out[2, i] = p1
        elif wi>=D2: #nằm bên phải sóng phải
            P_out[0, i] = r2
            P_out[1, i] = u2
            P_out[2, i] = p2
        elif D1star<= wi <= U: #nằm giữa hai sóng 
            P_out[0, i] = R1
            P_out[1, i] = U
            P_out[2, i] = P
        elif U<= wi <= D2star: #nằm giữa hai sóng 
            P_out[0, i] = R2
            P_out[1, i] = U
            P_out[2, i] = P
        elif D1< wi < D1star: #nằm trong sóng giãn trái
            cstar = gm1dgp1*(u1 - wi) + c1/gp1d2
            P_out[1, i] = wi + cstar
            P_out[2, i] = p1*(cstar/c1)**g2dgm1           
            P_out[0, i] = g*P_out[2, i]/cstar**2
        elif D2star< wi < D2: #nằm trong sóng giãn phải 
            cstar = gm1dgp1*(wi - u2) + c2/gp1d2
            P_out[1, i] = wi - cstar
            P_out[2, i] = p2*(cstar/c2)**g2dgm1          
            P_out[0, i] = g*P_out[2, i]/cstar**2
        else:
            print 'Decay Error!'

    return P_out

#tính bước thời gian 
def time_step(P, CFL, dx):
    r, u, p = P[:, 0], P[:, 1], P[:, 2]
    c = np.sqrt(g*p/r) #vận tốc âm thanh 
    u_max = max(abs(u) + c)        
    dt = CFL*dx/u_max
    return dt

#tái cấu trúc nghiệm hằng số từng mảnh - Godunov
def godunov_reconstr(Ps):
    P_left = Ps[:-1].copy()
    P_right = Ps[1:].copy()
    return P_left, P_right

def euler_solver(Ps, pressure_solver, reconstr, runge_kutta, x, time_target, CFL):
    nx = len(x)
    dx = x[1] - x[0] #xét lưới đều 

    Us = Ps.copy()
    P2U(Ps, Us)
    Pstar = Ps[:-1].copy()
    Fstar = Ps[:-1].copy()
    time = 0.0
    while(time < time_target):
        Un = Us.copy()
        #tìm dt
        dt = time_step(Ps, CFL, dx) 
        if(time+dt > time_target): dt = time + dt-time_target
        time += dt
        for stage in range(runge_kutta):
            #bước 1: tái cấu trúc - reconstruction 
            P_left, P_right = reconstr(Ps)
            #bước 2: tìm nghiệm phân rã gián đoạn, tính hàm dòng
            for i in range(nx-1): Pstar[i] = decay_godunov(P_left[i], P_right[i], pressure_solver)
            P2F(Pstar, Fstar)
                
            #bước 3: tích phân theo thời gian
            if stage == 0: Us[1:-1] = Un[1:-1] - dt/dx*(Fstar[1:] - Fstar[:-1])
            elif stage == 1: Us[1:-1] = 0.75*Un[1:-1] + 0.25*Us[1:-1] - 0.25*dt/dx*(Fstar[1:] - Fstar[:-1])
            else: Us[1:-1] = 1./3*Un[1:-1] + 2./3*Us[1:-1] - 2./3*dt/dx*(Fstar[1:] - Fstar[:-1])
            
            U2P(Us, Ps) #tìm biến nguyên thủy
            
            #điều kiện biên: outflow
            Ps[0] = Ps[1]
            Ps[-1] = Ps[-2]        
    return  Ps
