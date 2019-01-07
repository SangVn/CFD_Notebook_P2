#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 

from constants import *

#Giải bài toán phân rã gián đoạn trong trên từng bề mặt thể tích hữu hạn
#phân rã gián đoạn Godunov
def decay_godunov(Pl, Pr, pressure_solver):
    r1, u1, p1 = Pl[0], Pl[1], Pl[2]
    r2, u2, p2 = Pr[0], Pr[1], Pr[2]
    
    #tìm P, U, R
    P, c1, c2, a1, a2 = pressure_solver(Pl, Pr)
    U = (a1*u1 + a2*u2 + p1 - p2)/(a1 + a2)
    #xét gián đoạn bên trái 
    if P > p1: #nếu là sóng xung kích 
        D1 = u1 - a1/r1
        R1 = r1*a1/(a1-r1*(u1-U))
    else: #nếu là sóng giãn
        D1 = u1 - c1
        c1star = c1 + gm1d2*(u1-U)
        D1star = U - c1star
        R1 = g*P/c1star**2
    #tương tự cho gián đoạn bên phải 
    if P > p2: #nếu là sóng xung kích 
        D2 = u2 + a2/r2
        R2 = r2*a2/(a2 + r2*(u2-U))
    else: #nếu là sóng giãn
        D2 = u2 + c2
        c2star = c2 - gm1d2*(u2-U)
        D2star = U + c2star
        R2 = g*P/c2star**2
        
    #xét cấu hình phân rã xác định nghiệm PStar = (Rstar, Ustar, Pstar)
    #tùy theo vị trí biên i+1/2 nằm trong vùng nào (xem bài 12)

    if D1>0 and D2>0:   #nằm bên trái sóng trái
        Rstar = r1
        Ustar = u1
        Pstar = p1
    elif D1<0 and D2<0: #nằm bên phải sóng phải
        Rstar = r2
        Ustar = u2
        Pstar = p2
    elif D1<0 and D2>0: #nằm giữa hai sóng 
        if U>=0: Rstar = R1 #nằm bên trái gián đoạn tiếp xúc
        else:    Rstar = R2 #nằm bên phải gián đoạn tiếp xúc
        Ustar = U
        Pstar = P
    elif D1<0 and D1star>0: #nằm trong sóng giãn trái
        Ustar = gm1dgp1*u1 + c1/gp1d2
        Pstar = p1*(Ustar/c1)**g2dgm1
        Rstar = g*p1/Ustar**2   
    elif D2>0 and D2star<0: #nằm trong sóng giãn phải 
        Ustar = gm1dgp1*u2 - c2/gp1d2
        Pstar = p2*(Ustar/c2)**g2dgm1
        Rstar = g*p2/Ustar**2
    else:
        print 'Godunov _ Decay Error!'
        
    #vector biến gốc PStar
    PStar = [Rstar, Ustar, Pstar]
    return PStar
