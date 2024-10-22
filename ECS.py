import random
import math
import numpy as np
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import copy
import sys
df=pd.read_csv("data2.csv",delimiter=",")
Y=df["Target"]
X=df.iloc[:,0:9]
X=pd.get_dummies(X,columns=['LDU',"PT"])

X=np.array(X)
# scale_X=MinMaxScaler().fit_transform( X)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=42)
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# Fit the regressor with x and y data
regressor.fit(X_train, y_train)

predicted_value = regressor.predict(X_test)
 
# Evaluating the model
mse = mean_squared_error(y_test, predicted_value)
print(f'Mean Squared Error: {mse}')
 
r2 = r2_score(y_test, predicted_value)
print(f'R-squared: {r2}')

plt.figure(figsize=(10,10))
plt.scatter(y_test, predicted_value, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predicted_value), max(y_test))
p2 = min(min(predicted_value), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

"""
num of exist=400
num of zone=10
z1=80
z2=60
z3=15
z4=5
z5=30
z6=25
z7=35
z8=50
z9=70
z10=30
num of candidate=300
z1=20
z2=30
z3=35
z4=25
z5=30
z6=45
z7=35
z8=25
z9=35
z10=20
num of desired=50
"""

X_exist=X_test[0:400].copy()

#----------------PSO Algorithm----------------
def fitness_function(position):
  X_candidate=X_test[400:700].copy()
  global P
  sz1=0
  sz2=0
  sz3=0
  sz4=0
  sz5=0
  sz6=0
  sz7=0
  sz8=0
  sz9=0
  sz10=0
  P=[]
  for i,p in enumerate(position):
      # print("i",i,"p",p)
      if p>=.5 :
          P.append(1)
          if i in list(range(20)):
           sz1+=1  
          elif i in list(range(20,50)):
           sz2+=1
          elif i in list(range(50,85)):
           sz3+=1
          elif i in list(range(85,110)):
           sz4+=1
          elif i in list(range(110,140)):
           sz5+=1
          elif i in list(range(140,185)):
           sz6+=1
          elif i in list(range(185,220)):
           sz7+=1
          elif i in list(range(220,245)):
           sz8+=1
          elif i in list(range(245,280)):
           sz9+=1
          elif i in list(range(280,300)):
           sz10+=1
      else:
          P.append(0)
  sz=[sz1,sz2,sz3,sz4,sz5,sz6,sz7,sz8,sz9,sz10]  
  print(sz)
  mult1=[80/(80+sz1) for i in range(80)] 
  mult1.extend([60/(60+sz2) for i in range(60)] )
  mult1.extend([15/(15+sz3) for i in range(15)] )
  mult1.extend([5/(5+sz4) for i in range(5)] )
  mult1.extend([30/(30+sz5) for i in range(30)] )
  mult1.extend([25/(25+sz6) for i in range(25)] )
  mult1.extend([35/(35+sz7) for i in range(35)] )
  mult1.extend([50/(50+sz8) for i in range(50)] )
  mult1.extend([70/(70+sz9) for i in range(70)] )
  mult1.extend([30/(30+sz10) for i in range(30)] )
  
  mult2=[80/(80+sz1) for i in range(20)] 
  mult2.extend([60/(60+sz2) for i in range(30)] )
  mult2.extend([15/(15+sz3) for i in range(35)] )
  mult2.extend([5/(5+sz4) for i in range(25)] )
  mult2.extend([30/(30+sz5) for i in range(30)] )
  mult2.extend([25/(25+sz6) for i in range(45)] )
  mult2.extend([35/(35+sz7) for i in range(35)] )
  mult2.extend([50/(50+sz8) for i in range(25)] )
  mult2.extend([70/(70+sz9) for i in range(35)] )
  mult2.extend([30/(30+sz10) for i in range(20)] )
  # print(mult1)
  # print(mult2)  
  
  X_exist[:,1]=X_test[0:400,1]* mult1
  X_candidate[:,1]=X_test[400:700,1]* mult2  
  X_candidate=np.concatenate((X_candidate,np.expand_dims(np.array(P), axis=1)),axis=1)
  X_candidate=X_candidate[X_candidate[:,16]==1]
  X_candidate=X_candidate[:,0:16]
  fitness_value1=np.sum( regressor.predict(X_exist))
  fitness_value2=np.sum( regressor.predict(X_candidate))

  

  landa1=-1
  landa2=-1
  landa3=20
  landa4=10

  # print(fitness_value1,fitness_value2,sum(sz))
  fitness_value=landa1*fitness_value1+landa2*fitness_value2+landa3*((sum(sz)-50)**2)+landa4*(max(sz[0]-10,0)+max(sz[1]-10,0)+max(sz[2]-10,0)
                                                                                            +max(sz[3]-10,0)+max(sz[4]-10,0)+max(sz[5]-10,0)+max(sz[6]-10,0)+max(sz[7]-10,0)+max(sz[8]-10,0)+max(sz[9]-10,0))

 
  return fitness_value


print("-------------PSO Algorithm-------------------")
fitness_plt_PSO=[]
# fitness_plt_PSO.append(F)
class Particle:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
 
    # initialize position of the particle with 0.0 value
    self.position = [0.0 for i in range(dim)]
 
     # initialize velocity of the particle with 0.0 value
    self.velocity = [0.0 for i in range(dim)]
 
    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]
 
    # loop dim times to calculate random position and velocity
    # range of position and velocity is [minx, max]
    for i in range(dim):
      self.position[i] = ((maxx[i] - minx[i]) *
        self.rnd.random() + minx[i])
      self.velocity[i] = ((maxx[i] - minx[i]) *
        self.rnd.random() + minx[i])
 
    # compute fitness of particle
    self.fitness = fitness(self.position) # curr fitness
 
    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position) 
    self.best_part_fitnessVal = self.fitness # best fitness
 
# particle swarm optimization function
def pso(fitness, max_iter, n, dim, minx, maxx):
  # hyper parameters
  w = 0.729    # inertia
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
  global fitness_plt_PSO
  
  rnd = random.Random(0)
  global swarm
  # create n random particles
  swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)] 
 
  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best
 
  # computer best particle of swarm and it's fitness
  for i in range(n): # check each particle
    if swarm[i].fitness < best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position) 
 
  # main loop of pso
  Iter = 0
  while Iter < max_iter:
    print("Iter: ",Iter)
    # after every 10 iterations 
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
 
    for i in range(n): # process each particle
      # print("iiiii: ",i)  
      # compute new velocity of curr particle
      for k in range(dim): 
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
     
        swarm[i].velocity[k] = ( 
                                 (w * swarm[i].velocity[k]) +
                                 (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) + 
                                 (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k])) 
                               )  
 
 
        # if velocity[k] is not in [minx, max]
        # then clip it 

 
 
      # compute new position using new velocity
      for k in range(dim): 
        swarm[i].position[k] += swarm[i].velocity[k]
        # print("ss",swarm[i].position[k])
        if swarm[i].position[k] < minx[k]:
          swarm[i].position[k] = minx[k]
        elif swarm[i].position[k] > maxx[k]:
          swarm[i].position[k] = maxx[k]
      # compute fitness of new position
      # print("11111111111")
      swarm[i].fitness = fitness(swarm[i].position)
      # print("222222222")
      # is new position a new best for the particle?
      if swarm[i].fitness < swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)
 
      # is new position a new best overall?
      if swarm[i].fitness < best_swarm_fitnessVal:
        best_swarm_fitnessVal = swarm[i].fitness
        best_swarm_pos = copy.copy(swarm[i].position)
    
    fitness_plt_PSO.append(best_swarm_fitnessVal)
    
    # for-each particle
    Iter += 1
  #end_while
  return best_swarm_pos
# end pso
 
 
#----------------------------
# Driver code for rastrigin function
 


fitness = fitness_function
dim = 300
 
print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")


 
num_particles = 100
max_iter = 20
 
print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting PSO algorithm\n")
 
minx=list(np.zeros(300))
maxx=list(np.ones(300))
 


best_position = pso(fitness, max_iter, num_particles, dim, minx, maxx)
 
print("\nPSO completed\n")
print("\nBest solution found:")
# print(["%.6f"%best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution = %.6f" % fitnessVal)

print(P)
print("\nEnd particle swarm for rastrigin function\n")
 
plt.figure()
plt.plot(fitness_plt_PSO)
plt.title("Optimization Convergence")
plt.xlabel("Iterations")
plt.ylabel("Fitness Function")
print()
print()