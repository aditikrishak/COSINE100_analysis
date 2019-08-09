"""
Parameter estimation and Model comparison of cosine-100 experiment

@author: aditi
"""

import numpy as np 
from scipy import optimize , stats
import nestle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

path=input("enter file path ")    
t=open(path, "r")
data1 = np.loadtxt(t,delimiter=',')                    
t.close()
data1=np.transpose(data1)

path=input("enter file path ")    
t=open(path, "r")
data2 = np.loadtxt(t,delimiter=',')                    
t.close()
data2=np.transpose(data2)

path=input("enter file path ")    
t=open(path, "r")
data3 = np.loadtxt(t,delimiter=',')                    
t.close()
data3=np.transpose(data3)

path=input("enter file path ")    
t=open(path, "r")
data4 = np.loadtxt(t,delimiter=',')                    
t.close()
data4=np.transpose(data4)

path=input("enter file path ")    
t=open(path, "r")
data5 = np.loadtxt(t,delimiter=',')                    
t.close()
data5=np.transpose(data5)

data=np.hstack((data1,data2,data3,data4,data5))

def fit_bg(x,c,p0,p1):
    	return c + p0*np.exp(-np.log(2)*x/p1)

#background-only best fit (for each data set separately)
guess=[0,1,1000]
def chi2(P,DATA):
        sigma1=[ np.sqrt( (DATA[3,i]**2) + ((-P[1]*np.log(2)*np.exp(-np.log(2)*DATA[0,i]/P[2])/P[2])*DATA[2,i])**2 ) for i in range(len(DATA[0])) ]
        y_fit1=fit_bg(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit1)/sigma1
        return np.sum(r**2)

cos1 = optimize.minimize(chi2, guess,args=data1,method='BFGS')
cos2 = optimize.minimize(chi2, guess,args=data2,method='BFGS')
cos3 = optimize.minimize(chi2, guess,args=data3,method='BFGS')
cos4 = optimize.minimize(chi2, guess,args=data4,method='BFGS')
cos5 = optimize.minimize(chi2, guess,args=data5,method='BFGS')

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


""" prior transform as:  ub = upper bound , lb = lower bound
                         (ub-lb)*[some value b/w 0 and 1] + lb
                         at 0, prior = lb
                         at 1, prior = ub"""
#5% interval around value - returns lower bound

def b(m):
    if m>=0.0: return 0.95*m
    else: return 1.05*m

#5% interval around value - returns diff between upper and lower bound
def d(m):
    return np.abs(0.1*m)

def prior_transform_bg(P):
    return np.array([d(cos1.x[0])*P[0]+b(cos1.x[0]) , d(cos1.x[1])*P[1]+b(cos1.x[1]) , 
                     d(cos1.x[2])*P[2]+b(cos1.x[2])  ,d(cos2.x[0])*P[3]+b(cos2.x[0]), 
                     d(cos2.x[1])*P[4]+b(cos2.x[1]), d(cos2.x[2])*P[5]+b(cos2.x[2]), 
                     d(cos3.x[0])*P[6]+b(cos3.x[0]), d(cos3.x[1])*P[7]+b(cos3.x[1]), 
                     d(cos3.x[2])*P[8]+b(cos3.x[2]) , d(cos4.x[0])*P[9]+b(cos4.x[0]) ,
                     d(cos4.x[1])*P[10]+b(cos4.x[1]), d(cos4.x[2])*P[11]+b(cos4.x[2]),   
                     d(cos5.x[0])*P[12]+b(cos5.x[0]), d(cos5.x[1])*P[13]+b(cos5.x[1]) ,
                     d(cos5.x[2])*P[14]+b(cos5.x[2])])
    

def nestle_multi_bg():
    # Run nested sampling
    res = nestle.sample(log_likelihood_bg, prior_transform_bg, 15, 
                        method='multi', npoints=2000)
    print(res.summary())
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)
    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    FIT=np.array([np.mean(samples_nestle[:,i]) for i in range(15)])
    return res.logz , FIT
#----------------------------------------------------------------------------
def fit_cosine(x,A,w,t_0,c,p0,p1):
    	return c + p0*np.exp(-np.log(2)*x/p1) + A*np.cos(w*(x-t_0))

def chi2_cosine(P):
    A = P[0]
    w = P[1]
    t_0= P[2]
    
    sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[5])/P[5] - A*w*np.sin(w*(data1[0][i]-t_0))) *data1[2,i])**2 ) for i in range(len(data1[0])) ]
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[3],P[4],P[5])
    
    sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[8])/P[8] - A*w*np.sin(w*(data2[0][i]-t_0))) *data2[2,i])**2 ) for i in range(len(data2[0])) ]    
    y_fit2=fit_cosine(data2[0],P[0],P[1],P[2],P[6],P[7],P[8])
    
    sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[11])/P[11] - A*w*np.sin(w*(data3[0][i]-t_0))) *data3[2,i])**2 ) for i in range(len(data3[0])) ]
    y_fit3=fit_cosine(data3[0],P[0],P[1],P[2],P[9],P[10],P[11])
    
    sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[14])/P[14] - A*w*np.sin(w*(data4[0][i]-t_0))) *data4[2,i])**2 ) for i in range(len(data4[0])) ]
    y_fit4=fit_cosine(data4[0],P[0],P[1],P[2],P[12],P[13],P[14])
    
    sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[16]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[17])/P[17] - A*w*np.sin(w*(data5[0][i]-t_0))) *data5[2,i])**2 ) for i in range(len(data5[0])) ]
    y_fit5=fit_cosine(data5[0],P[0],P[1],P[2],P[15],P[16],P[17])
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    r = (data[1] - y_fit)/sigma
    return np.sum(r**2)

def log_likelihood_cosine(P):
    A=P[0]
    w=P[1]
    t_0=P[2]
    
    sigma1=[ np.sqrt( (data1[3,i]**2) + ((-P[4]*np.log(2)*np.exp(-np.log(2)*data1[0,i]/P[5])/P[5] - A*w*np.sin(w*(data1[0][i]-t_0))) *data1[2,i])**2 ) for i in range(len(data1[0])) ]
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[3],P[4],P[5])
    
    sigma2=[ np.sqrt( (data2[3,i]**2) + ((-P[7]*np.log(2)*np.exp(-np.log(2)*data2[0,i]/P[8])/P[8] - A*w*np.sin(w*(data2[0][i]-t_0))) *data2[2,i])**2 ) for i in range(len(data2[0])) ]    
    y_fit2=fit_cosine(data2[0],P[0],P[1],P[2],P[6],P[7],P[8])
    
    sigma3=[ np.sqrt( (data3[3,i]**2) + ((-P[10]*np.log(2)*np.exp(-np.log(2)*data3[0,i]/P[11])/P[11] - A*w*np.sin(w*(data3[0][i]-t_0))) *data3[2,i])**2 ) for i in range(len(data3[0])) ]
    y_fit3=fit_cosine(data3[0],P[0],P[1],P[2],P[9],P[10],P[11])
    
    sigma4=[ np.sqrt( (data4[3,i]**2) + ((-P[13]*np.log(2)*np.exp(-np.log(2)*data4[0,i]/P[14])/P[14] - A*w*np.sin(w*(data4[0][i]-t_0))) *data4[2,i])**2 ) for i in range(len(data4[0])) ]
    y_fit4=fit_cosine(data4[0],P[0],P[1],P[2],P[12],P[13],P[14])
    
    sigma5=[ np.sqrt( (data5[3,i]**2) + ((-P[16]*np.log(2)*np.exp(-np.log(2)*data5[0,i]/P[17])/P[17] - A*w*np.sin(w*(data5[0][i]-t_0))) *data5[2,i])**2 ) for i in range(len(data5[0])) ]
    y_fit5=fit_cosine(data5[0],P[0],P[1],P[2],P[15],P[16],P[17])
    
    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))

def prior_transform_cos(P):
    return np.array([0.005*P[0]+0.005 ,(0.01*2.0*np.pi/365.25)*P[1]+0.99*(2.0*np.pi/365.25) , 100*P[2]+80 , 
                     d(cos1.x[0])*P[3]+b(cos1.x[0]) , d(cos1.x[1])*P[4]+b(cos1.x[1]) ,
                     d(cos1.x[2])*P[5]+b(cos1.x[2])  ,
                     d(cos2.x[0])*P[6]+b(cos2.x[0]), d(cos2.x[1])*P[7]+b(cos2.x[1]),
                     d(cos2.x[2])*P[8]+b(cos2.x[2]), 
                     d(cos3.x[0])*P[9]+b(cos3.x[0]), d(cos3.x[1])*P[10]+b(cos3.x[1]), 
                     d(cos3.x[2])*P[11]+b(cos3.x[2]) , 
                     d(cos4.x[0])*P[12]+b(cos4.x[0]), d(cos4.x[1])*P[13]+b(cos4.x[1]), 
                     d(cos4.x[2])*P[14]+b(cos4.x[2]),   
                     d(cos5.x[0])*P[15]+b(cos5.x[0]), d(cos5.x[1])*P[16]+b(cos5.x[1]),
                     d(cos5.x[2])*P[17]+b(cos5.x[2])])


def nestle_multi_cos():
    # Run nested sampling
    res = nestle.sample(log_likelihood_cosine, prior_transform_cos, 18, 
                        method='multi', npoints=2000)
    print(res.summary())
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    # re-scale weights to have a maximum of one
    nweights = res.weights/np.max(res.weights)
    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    samples_nestle = res.samples[keepidx,:]
    # weighted average and covariance:
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    FIT=np.array([np.mean(samples_nestle[:,i]) for i in range(18)])
    return res.logz , FIT
#-----------------------------------------------------------------------------
def frequentist(cos_fin,k_fin):
    c1=chi2_cosine(cos_fin)
    c2=chi2_bg(k_fin)
    d=np.abs(c1-c2)
    print("difference in chi square values = ",d)
    p=stats.chi2(3).sf(d)
    print ("p value=",p)
    print("Confidence level : ",stats.norm.isf(p),'\u03C3','\n')
    
def AIC(cos_fin,k_fin):
    aic_bg=-2*log_likelihood_bg(k_fin) + 2*15
    aic_cosine=-2*log_likelihood_cosine(cos_fin) +2*18
    del_aic= np.abs(aic_bg-aic_cosine)
    print("AIC cosine=",'%.2f'%aic_cosine,"AIC bg=",'%.2f'%aic_bg)
    print ("diff in AIC values = ",'%.2f'%del_aic)
    
def BIC(cos_fin,k_fin):
    bic_bg=-2*log_likelihood_bg(k_fin) + 15*np.log(len(data[0]))
    bic_cosine=-2*log_likelihood_cosine(cos_fin) +18*np.log(len(data[0]))
    del_bic= np.abs(bic_bg-bic_cosine)
    print("BIC cosine=",'%.2f'%bic_cosine,"BIC bg=",'%.2f'%bic_bg)
    print ("diff in BIC values = ",'%.2f'%del_bic)

def bayesian(Zbg,Zcos):  
    Z= np.exp(Zcos-Zbg)
    print('Bayes Factor: ',Z)
#-----------------------------------------------------------------------------
def plot(bg_fit, cosine_fit):
       
    fig,ax=plt.subplots(nrows=5,ncols=1,figsize=(15,15))
    
    
    plt.subplot(511)
    p1 = np.linspace(data1[0].min(),data1[0].max(),10000)   
    plt.plot(p1, fit_cosine(p1,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                  cosine_fit[3],cosine_fit[4],cosine_fit[5]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data1[0],data1[1],c='black',s=20)
    plt.plot(p1, np.ones(10000)*fit_bg(p1,bg_fit[0],bg_fit[1],bg_fit[2]),
                color='dodgerblue',lw=1,linestyle='--',label='$H_0$' ,linewidth=1.75)
    plt.grid(color='w')
    plt.errorbar(data1[0],data1[1],yerr = data1[3],xerr=data1[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.25,"Crystal 2",fontsize=15)
    
    
    plt.subplot(512)
    p2 = np.linspace(data2[0].min(),data2[0].max(),10000)
    plt.plot(p2, fit_cosine(p2,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                        cosine_fit[6],cosine_fit[7],cosine_fit[8]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data2[0],data2[1],c='black',s=20)
    plt.plot(p2, np.ones(10000)*fit_bg(p2,bg_fit[3],bg_fit[4],bg_fit[5]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data2[0],data2[1],yerr = data2[3],xerr=data2[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.55,"Crystal 3",fontsize=15)
    
        
    plt.subplot(513)
    p3 = np.linspace(data3[0].min(),data3[0].max(),10000)
    plt.plot(p3, fit_cosine(p3,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                    cosine_fit[9],cosine_fit[10],cosine_fit[11]), color = 'red' ,label='$H_1$',linewidth=1.6)
    plt.scatter(data3[0],data3[1],c='black',s=20)
    plt.plot(p3, np.ones(10000)*fit_bg(p3,bg_fit[6],bg_fit[7],bg_fit[8]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data3[0],data3[1],yerr = data3[3],xerr=data3[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.5,"Crystal 4",fontsize=15)
    
        
    plt.subplot(514)
    p4 = np.linspace(data4[0].min(),data4[0].max(),10000)
    plt.plot(p4, fit_cosine(p4,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                    cosine_fit[12],cosine_fit[13],cosine_fit[14]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data4[0],data4[1],c='black',s=20)
    plt.plot(p4, np.ones(10000)*fit_bg(p4,bg_fit[9],bg_fit[10],bg_fit[11]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data4[0],data4[1],yerr = data4[3],xerr=data4[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,2.7,"Crystal 6",fontsize=15)
    
    
    plt.subplot(515)
    p5 = np.linspace(data5[0].min(),data5[0].max(),10000)
    plt.plot(p5, fit_cosine(p5,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                    cosine_fit[15],cosine_fit[16],cosine_fit[17]), color = 'red',label='$H_1$',linewidth=1.6 )
    plt.scatter(data5[0],data5[1],c='black',s=20)
    plt.plot(p5, np.ones(10000)*fit_bg(p5,bg_fit[12],bg_fit[13],bg_fit[14]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data5[0],data5[1],yerr = data5[3],xerr=data5[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,2.75,"Crystal 7",fontsize=15)
    
    
    fig.text(0.5, 0.085, '\nTime (days)', ha='center',fontweight='bold',color='dimgrey',fontsize=17)
    fig.text(0.075, 0.5, '2-6 keV event rate(cpd/kg/keV)',va='center',rotation='vertical',fontweight='bold',fontsize=17,color='dimgrey')
        
    plt.savefig("cosinefigure.png")
#============================================================================
    
#background-only fit by chi-sq minimization
bg_est=np.hstack((cos1.x,cos2.x,cos3.x,cos4.x,cos5.x))

# background only fits using nestle
Z1, bg_fit = nestle_multi_bg()

# cosine fits using nestle
Z2, cosine_fit = nestle_multi_cos()

plot(bg_est, cosine_fit)

# Model Comparison
frequentist(cosine_fit,bg_est)

AIC(cosine_fit,bg_est)

BIC(cosine_fit,bg_est)

bayesian(Z1,Z2)
