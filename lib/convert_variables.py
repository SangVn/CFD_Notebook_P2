#!/usr/bin/env python
# coding: utf-8
# Copyright (C) 2018  Nguyen Ngoc Sang, <https://github.com/SangVn> 

from constants import gm1, gdgm1

#hàm chuyển đổi từ P sang U 
def P2U(P, U):
    r, u, p = P[:, 0], P[:, 1], P[:, 2]
    U[:, 0] = r
    U[:, 1] = r*u
    U[:, 2] = r*u**2/2. + p/gm1
    return

#hàm chuyển đổi từ U về P
def U2P(U, P):
    r, ru, re = U[:, 0], U[:, 1], U[:, 2]
    P[:, 0] = r
    P[:, 1] = u = ru/r
    P[:, 2] = (re - ru*u/2.)*gm1
    return

#hàm chuyển đổi từ P sang F
def P2F(P, F):
    r, u, p = P[:, 0], P[:, 1], P[:, 2]
    F[:, 0] = r*u
    F[:, 1] = r*u**2 + p
    F[:, 2] = r*u**3/2. + p*u*gdgm1
    return
