""""
                                 ************charging demand prediction features*****************
Response Variable: 

f1: The number of charging events of CS 𝑖 on week 𝑤                              /Numerical 

Spatial Context Information:
    
f2: The density of charging stations in the same zip code area of CS location 𝑖  / Numerical 
f3: Land use type: if CS location 𝑖 is in an institutional area                   /Binary 
f4: Land use type: if CS location 𝑖 is in a transportation area                   /Binary 
f5: Land use type: if CS location 𝑖 is in a commercial area                       /Binary 
f6: Land use type: if CS location 𝑖 is in a residential area                      /Binary 
f7: Land use type: if CS location 𝑖 is in a recreational area                     /Binary 
f8: Land use type: if CS location 𝑖 is in a vacant area                           /Binary 
f9: Land use type: if CS location 𝑖 is in an industrial area                      /Binary 

Weather Information :

f10: Weekly precipitation of week 𝑤 (𝑚𝑚)                                 /Numerical 
f11: Weekly average temperature of week 𝑤 (°𝐶)                            /Numerical 
f12: Weekly average wind speed of week 𝑤 (𝑚/𝑠𝑒𝑐)                          /Numerical 

Charger Type :
    
f13: 𝑇1 Port type of DC fast charger for CS location 𝑖                      /Binary 
f14: 𝑇2 Port type of level 2 charger for CS location 𝑖                      /Binary 

Traffic Information:
    
f15: Annual average daily traffic on CS location 𝑖’s nearby roads           /Numerical 
f16: Trip production of the TAZ where CS location 𝑖 is located              /Numerical 

"""
import random
import math
import numpy as np
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
N=30000
def pwm(f1,f2,f3,n):
    noise=np.random.rand(N)-.5
    return (signal.square(2 * np.pi * f1 * t, duty=(sig + 1)/2)+ signal.square(2 * np.pi * f2 * t, duty=(sig + 1)/2)+signal.square(2 * np.pi * f3 * t, duty=(sig + 1)/2))/3+n*noise
from sklearn.preprocessing import MinMaxScaler
l1=["LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU2","LDU2",
   "LDU2","LDU2","LDU2","LDU2","LDU2","LDU2","LDU3","LDU3","LDU3","LDU3","LDU3","LDU3","LDU3","LDU3",
   "LDU4","LDU4","LDU4","LDU4","LDU5","LDU5","LDU5","LDU6","LDU6","LDU6","LDU7","LDU7"]

l2=["DC","DC","DC","DC","DC","DC","DC","DC","l2","l2"]
ll1=l1.copy()
ll2=l2.copy()
for i in range(809):    
    ll1.extend(l1)
ll1.extend(["LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1"
            ,"LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1","LDU1"])
l1=ll1
for i in range(2999):    
    ll2.extend(l2)
l2=ll2


t = np.linspace(0, 1, N, endpoint=False)
noise=np.random.rand(N)-.5
sig = np.sin(2 * np.pi * t)

random.shuffle(l1)
random.shuffle(l2)

f1=1000+np.random.randn(N)*100
l=list(7+np.random.randn(10000)*1)
l.extend(list(20+np.random.randn(10000)*3))
l.extend(list(70+np.random.randn(10000)*10))
f2=np.array(l)
f3=np.array(l1)
f13=np.array(random.shuffle(l1))
f10=90+np.random.randn(N)*10
f11=pwm(20,40,60,.05)*6+25
f12=(pwm(60,80,120,.05)*6+20)/3
f13=l2
f15=20000+np.random.randn(N)*5000+50*noise
f16=2000+np.random.randn(N)*500+10*noise


data=np.stack((f1,f2,f3,f10,f11,f12,f13,f15,f16),axis=1)

df=pd.DataFrame(data,columns=["D","NHCS","LDU","PRCP","TMP","WD","PT","AADT","TP"])
#df2=df.drop(['LDU',"PT"],axis=1)

df1=pd.get_dummies(df,columns=['LDU',"PT"])
arr=np.array(df1)
scale_arr=MinMaxScaler().fit_transform( arr)
y=100*scale_arr[:,0]*scale_arr[:,1]+100*(scale_arr[:,0]+scale_arr[:,1])**3+120*scale_arr[:,1]*np.sin(scale_arr[:,1])+50*np.sinc(scale_arr[:,1])+5*scale_arr[:,1]+10*scale_arr[:,1]**5+.3*scale_arr[:,1]**2+3*scale_arr[:,2]*scale_arr[:,1]+4*scale_arr[:,3]+(scale_arr[:,3]+scale_arr[:,1])**2+5*scale_arr[:,4]**3
+2*scale_arr[:,5]*scale_arr[:,1]+1*scale_arr[:,6]+7*scale_arr[:,7]+2.5*scale_arr[:,8]+4*scale_arr[:,9]+.1*scale_arr[:,10]+5*scale_arr[:,11]-4.3*scale_arr[:,12]
+10*scale_arr[:,13]+30*(scale_arr[:,14]-scale_arr[:,1])**4-10*scale_arr[:,15]+80*np.sin(scale_arr[:,1]*scale_arr[:,3]*scale_arr[:,4])+10*noise

df["Target"]=y
df.to_csv("data2.csv",sep=',',index=False)
