#Bayesian analysis : all parameters free
# This code computes the Bayesian evidence or marginal likelihood for background +cosine model with all parameters free

import numpy as np 
from scipy import optimize , stats
import os
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
import nestle
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from contextlib import closing
plt.style.use('ggplot')
tol=0.1

f=input('enter path to data file : ')
data1 = np.loadtxt(f,delimiter=',')                    

g=input('enter path to data file : ')
data2 = np.loadtxt(g,delimiter=',')                    

h=input('enter path to data file : ')
data3 = np.loadtxt(h,delimiter=',')                    

k=input('enter path to data file : ')
data4 = np.loadtxt(k,delimiter=',')                    

l=input('enter path to data file : ')
data5 = np.loadtxt(l,delimiter=',')                    

data=np.hstack((data1,data2,data3,data4,data5))


def fit_cosine(x,c,p0,p1,A,w,t_0):
    	return c + p0*np.exp(-np.log(2)*x/p1) + A*np.cos(w*(x-t_0))

def log_likelihood_cosine(P):
    A = P[15]
    w = P[16]
    t_0=P[17]
    sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[1]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[2])/P[2] - A*w*np.sin(w*(data1[0][i]-t_0))) *data1[2,i])**2 ) for i in range(len(data1[0])) ]
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[15],P[16],P[17])
    
    sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[5])/P[5] - A*w*np.sin(w*(data2[0][i]-t_0))) *data2[2,i])**2 ) for i in range(len(data2[0])) ]    
    y_fit2=fit_cosine(data2[0],P[3],P[4],P[5],P[15],P[16],P[17])
    
    sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[8])/P[8] - A*w*np.sin(w*(data3[0][i]-t_0))) *data3[2,i])**2 ) for i in range(len(data3[0])) ]
    y_fit3=fit_cosine(data3[0],P[6],P[7],P[8],P[15],P[16],P[17])
    
    sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[11])/P[11] - A*w*np.sin(w*(data4[0][i]-t_0))) *data4[2,i])**2 ) for i in range(len(data4[0])) ]
    y_fit4=fit_cosine(data4[0],P[9],P[10],P[11],P[15],P[16],P[17])
    
    sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[14])/P[14] - A*w*np.sin(w*(data5[0][i]-t_0))) *data5[2,i])**2 ) for i in range(len(data5[0])) ]
    y_fit5=fit_cosine(data5[0],P[12],P[13],P[14],P[15],P[16],P[17])
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))
    

a1=100.0*np.max(data[1])
b1=30000.0
#def prior_transform_cos(P):
#        return np.array([P[0]*a,P[1]*a,P[2]*b,P[3]*a,P[4]*a,P[5]*b,P[6]*a,P[7]*a,P[8]*b,P[9]*a,P[10]*a,P[11]*b,P[12]*a,P[13]*a,P[14]*b,P[15]*a,P[16]*a,P[17]*a])

def prior_transform_cos(P):
        return np.array([(a1+100)*P[0]-100,a1*P[1],b1*P[2],P[3]*(a1+100)-100,a1*P[4],P[5]*b1,(a1+100)*P[6]-100,P[7]*a1,P[8]*b1,P[9]*(a1+100)-100,P[10]*a1,P[11]*b1,P[12]*(a1+100)-100,P[13]*a1,P[14]*b1,P[15]*(a1+100)-100,P[16]*6.2,P[17]*361.0])


def nestle_multi_cos():
    # Run nested sampling
#    sampler = DynamicNestedSampler(log_likelihood_cosine, prior_transform_cos, 18, 
#                        bound='multi',nlive=1024,sample='rwalk')    
#        with Pool() as pool:
        with closing(Pool(processes=24)) as pool:
                sampler = NestedSampler(log_likelihood_cosine, prior_transform_cos, 18, 
                                        bound='balls',nlive=1024,sample='rwalk',pool=pool,queue_size=24)
                t0 = time.time()
                sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
                t0 = time.time()
                pool.terminate
        res=sampler.results
        print (res.summary())
        print( res.logz)
        return res.logz[-1], res.logzerr[-1]
print( "c_f_dynasty_noprior calling nestle_multi_cos : background priors and  signal priors completely free : integrating w to 1 day : a1 changed to 10000: used priors on C less than 0: everything else > 0. bound=balls b1=30000")
Z2,Z2err = nestle_multi_cos()
print( "after nestlemulti_cos")
print(Z2)
print (Z2err)
