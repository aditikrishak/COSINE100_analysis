# Bayesian Analysis : Background only

import numpy as np 
from scipy import optimize , stats
import os
import dynesty
import time
from dynesty import NestedSampler
from multiprocessing import Pool
from contextlib import closing
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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

tol=0.1

def fit_bg(x,c,p0,p1):
    	return c + p0*np.exp(-np.log(2)*x/p1)

def chi2_bg(P):
    
        sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[1]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[2])/P[2])*data1[2,i])**2 ) for i in range(len(data1[0])) ]
        y_fit1=fit_bg(data1[0],P[0],P[1],P[2])
        
        sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[5])/P[5])*data2[2,i])**2 ) for i in range(len(data2[0])) ]
        y_fit2=fit_bg(data2[0],P[3],P[4],P[5])
        
        sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[8])/P[8])*data3[2,i])**2 ) for i in range(len(data3[0])) ]
        y_fit3=fit_bg(data3[0],P[6],P[7],P[8])
        
        sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[11])/P[11])*data4[2,i])**2 ) for i in range(len(data4[0])) ]
        y_fit4=fit_bg(data4[0],P[9],P[10],P[11])
        
        sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[14])/P[14])*data5[2,i])**2 ) for i in range(len(data5[0])) ]
        y_fit5=fit_bg(data5[0],P[12],P[13],P[14])    
        
        y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
        sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))        

        r=(data[1]-y_fit)/sigma
        return np.sum(r**2)

def log_likelihood_bg(P):
    
        sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[1]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[2])/P[2])*data1[2,i])**2 ) for i in range(len(data1[0])) ]
        y_fit1=fit_bg(data1[0],P[0],P[1],P[2])
        
        sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[5])/P[5])*data2[2,i])**2 ) for i in range(len(data2[0])) ]
        y_fit2=fit_bg(data2[0],P[3],P[4],P[5])
        
        sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[8])/P[8])*data3[2,i])**2 ) for i in range(len(data3[0])) ]
        y_fit3=fit_bg(data3[0],P[6],P[7],P[8])
        
        sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[11])/P[11])*data4[2,i])**2 ) for i in range(len(data4[0])) ]
        y_fit4=fit_bg(data4[0],P[9],P[10],P[11])
        
        sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[14])/P[14])*data5[2,i])**2 ) for i in range(len(data5[0])) ]
        y_fit5=fit_bg(data5[0],P[12],P[13],P[14])    

        y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
        sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))

        return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))


a1=100.0*np.max(data[1])
b1=30000.0

    

def prior_transform_bgnoprior(P):
        return np.array([(a1+100)*P[0]-100,a1*P[1],b1*P[2],P[3]*(a1+100)-100,a1*P[4],P[5]*b1,(a1+100)*P[6]-100,P[7]*a1,P[8]*b1,P[9]*(a1+100)-100,P[10]*a1,P[11]*b1,P[12]*(a1+100)-100,P[13]*a1,P[14]*b1])




def nestle_multi_bg():
        with closing(Pool(processes=24)) as pool:
    # Run nested sampling
                sampler = NestedSampler(log_likelihood_bg, prior_transform_bgnoprior, 15, 
                                        bound='balls', nlive=1024,sample='rwalk',pool=pool,queue_size=24)
                t0 = time.time()
                sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
                t1 = time.time()
                pool.terminate
        res=sampler.results
        print(res.summary())
        return res.logz[-1],res.logzerr[-1]



print ("using no priors on background model: a1 changed to 10000. priors starting from negative values for C. p0 & p1 been positive. bound=balls")
Z1,Z1err = nestle_multi_bg()
print (Z1)
print (Z1err)

