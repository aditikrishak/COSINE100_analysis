# Bayesian analysis : time period fixed
#this code calculates Bayesian evidence for cosine signal+background with w fixed


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



a1=100.0*np.max(data[1])
b1=30000.0

w=0.0172


def fit_cosine(x,c,p0,p1,A,t_0):
    	return c + p0*np.exp(-np.log(2)*x/p1) + A*np.cos(w*(x-t_0))



def log_likelihood_cosine(P):
    A = P[15]
    t_0=P[16]
    sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[1]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[2])/P[2] - A*w*np.sin(w*(data1[0][i]-t_0))) *data1[2,i])**2 ) for i in range(len(data1[0])) ]
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[15],P[16])
    
    sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[5])/P[5] - A*w*np.sin(w*(data2[0][i]-t_0))) *data2[2,i])**2 ) for i in range(len(data2[0])) ]    
    y_fit2=fit_cosine(data2[0],P[3],P[4],P[5],P[15],P[16])
    
    sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[8])/P[8] - A*w*np.sin(w*(data3[0][i]-t_0))) *data3[2,i])**2 ) for i in range(len(data3[0])) ]
    y_fit3=fit_cosine(data3[0],P[6],P[7],P[8],P[15],P[16])
    
    sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[11])/P[11] - A*w*np.sin(w*(data4[0][i]-t_0))) *data4[2,i])**2 ) for i in range(len(data4[0])) ]
    y_fit4=fit_cosine(data4[0],P[9],P[10],P[11],P[15],P[16])
    
    sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[14])/P[14] - A*w*np.sin(w*(data5[0][i]-t_0))) *data5[2,i])**2 ) for i in range(len(data5[0])) ]
    y_fit5=fit_cosine(data5[0],P[12],P[13],P[14],P[15],P[16])
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))

def prior_transform_cos(P):
        return np.array([(a1+100)*P[0]-100,a1*P[1],b1*P[2],P[3]*(a1+100)-100,a1*P[4],P[5]*b1,(a1+100)*P[6]-100,P[7]*a1,P[8]*b1,P[9]*(a1+100)-100,P[10]*a1,P[11]*b1,P[12]*(a1+100)-100,P[13]*a1,P[14]*b1,P[15]*(a1+100)-100,P[16]*361.0])



def nestle_multi_cos():
        with closing(Pool(processes=24)) as pool:
    # Run nestedsampling
                sampler = NestedSampler(log_likelihood_cosine, prior_transform_cos, 17, 
                            bound='balls', nlive=1024,sample='rwalk',pool=pool,queue_size=24)
#                t0 = time.time()
                sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
#                t1 = time.time()
                pool.terminate
        res=sampler.results
        print (res.summary())
        print(res.logz)
        return res.logz[-1],res.logzerr[-1]

#-----------------------------------------------------------------------------


print ("on signal+background (with w fixed): no priors on background or signal. Priors on C and signal amplitude start at negative values. everything else > 0 . using bound=balls. tol=0.1b1 changedto 30000")
Z1,Z1err = nestle_multi_cos()
print (Z1,Z1err)
