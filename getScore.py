import time
import sys
import random
import csv
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
import subprocess

def pywin(dperc):
	return 1.0/(1.0+((1.01-dperc)/(dperc+.01))**2.5)
def undopywin(x):
	a = (1.0/x-1.0)**(1.0/2.5)
	return (1.01-.01*a)/(a+1)
	
def writecsv(parr, filen):
        with open(filen, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(0,len(parr)):
                        try:
                                spamwriter.writerow(parr[i])
                        except:
                                print(parr[i], i)



def readcsv(filen):
        allgamesa  =[]
        with open(filen, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for row in spamreader:
                        allgamesa.append(row)
        return allgamesa
        

def isnmt(x):
	if x%10==9:
		return 1.1
	else:
		return 1.0
# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r,c: np.sum([(isnmt(p))*np.linalg.norm(c[r[p+1]]-c[r[p]]) for p in range(len(r)-1)])
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
	route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
	improvement_factor = 1 # Initialize the improvement factor.
	best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
	while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
		distance_to_beat = best_distance # Record the distance at the beginning of the loop.
		for swap_first in range(1,len(route)-3):
			for swap_last in range(swap_first+1,len(route)-1):
				new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
				new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
				if new_distance < best_distance: # If the path distance is an improvement,
					route = new_route # make this the accepted best route
					best_distance = new_distance # and update the distance corresponding to this route.
		improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
	return route
	
def read_tour(filename):
    tour = open(filename).read().split()[1:]
    tour = list(map(int, tour))
    if tour[-1] == 0: tour.pop()
    return tour

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    primes = list(sympy.primerange(0, len(cities)))
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

def write_submission(tour, filename):
    assert set(tour) == set(range(len(tour)))
    pd.DataFrame({'Path': list(tour) + [0]}).to_csv(filename, index=False)
    
cities = pd.read_csv('cities.csv', index_col=['CityId'])

for i in range(0,0):
	subprocess.run("./linkern -s "+str(10+i)+" -S linkern.tour -R 1000000000 -t 60 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))

for i in range(0,0):
	subprocess.run("./linkern -s "+str(20+i)+" -S linkern.tour -R 1000000000 -t 120 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))

for i in range(0,0):
	subprocess.run("./linkern -s "+str(30+i)+" -S linkern.tour -R 1000000000 -t 180 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))

for i in range(0,0):
	subprocess.run("./linkern -s "+str(40+i)+" -S linkern.tour -R 1000000000 -t 240 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))

for i in range(0,10):
	subprocess.run("./linkern -s "+str(50+i)+" -S linkern.tour -R 1000000000 -t 300 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))
	
for i in range(0,10):
	subprocess.run("./linkern -s "+str(60+i)+" -S linkern.tour -R 1000000000 -t 360 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkern.tour')
	print(score_tour(tour))

