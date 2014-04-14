import math
import numpy as np
import pdb
from pylab import *
import random


def imagesc(img):
    imshow(img, interpolation='nearest',aspect='auto' )
    #colorbar()

def fuse(p1, p2):  #fuse two vectors of probabilities (eg pi and lam for a set of nodes)
    q_on = p1*p2
    q_off = (1-p1)*(1-p2)
    return q_on / (q_on+q_off)

def applyTemperature(p, T):
    q_on = p**T
    q_off = (1-p)**T
    return q_on/(q_on+q_off)

def sanitiseTheta(th):
#ensures th is in interval:  (-np.pi,np.pi]
    while th<=-np.pi:
        th+=2*np.pi
    while th>np.pi:
        th-=2*np.pi
    return th

def lag(M, n):
    L = np.zeros(M.shape)
    L[n:, :] = M[0:-n]
    
    for t in range(0,n):
        L[t,:]=M[0,:] #cheat at t=0
    return L


def sig(x):
    return 1.0/(1.0 + np.exp(-x))

def invsig(x):
    return -np.log((1.0/x)-1)



#def gyroscope(hds_old, hds_new): 
#    th_old_x = hds_old[0]
#    th_old_y = hds_old[1]
#    th_new_x = hds_new[0]
#    th_new_y = hds_new[1]#
#    th_old = np.arctan2(th_#old_y, th_old_x)
#    th_new = np.arctan2(th_new_y, th_new_x)
#    d_th = sanitiseTheta(th_new - th_old)
#    return d_th



#crop to [xmin,xmax]  (inclusive)
def crop(x, xmin, xmax):
    if x<xmin:
        x=xmin
    if x>xmax:
        x=xmax
    return x

#thxys is an N*2 array, of x and y parts of theta at each time
#converts to int id of HD cell
#def thxy2ihds(thxys):
#    ihds = np.zeros((thxys.shape[0]))
#    for t in range(0, thxys.shape[0]):
#        if thxys[t,0]==1:
#            ihds[t]=0
#        if thxys[t,0]==-1:
#            ihds[t]=2
#        if thxys[t,1]==1:
#            ihds[t]=1
#        if thxys[t,1]==-1:
#            ihds[t]=3
#    return ihds

#convert head dir cells to their indices
#def hd2ihd(hd):
#    ihds = np.zeros((v_hds.shape[0]))
#    for t in range(0, v_hds.shape[0]):
#        ihds[t]=argmax(v_hds[t,:])
#    return ihds



#mean and stdev of beta distribution, starting from flat prior, after observing k heads from N coinflips. 
def cf_beta(N,k):
    alpha=1.0
    beta =1.0
    alpha += k
    beta += N-k
    mu = alpha/(alpha+beta)
    sigma = np.sqrt( (alpha*beta)/( (alpha+beta)**2 * (alpha+beta+1) ) )
    return (mu,sigma)
