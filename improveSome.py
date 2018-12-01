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

def read_tour(filename,nl,nm):
	if nm > 0:
		tour = open(filename).read().split()[1+nl:nm]
		tour = list(map(int, tour))
		if tour[-1] == 0: tour.pop()
	elif nl > 0:
		tour = open(filename).read().split()[1:]
		tour = list(map(int, tour))
		if tour[-1] == 0: tour.pop()
	else:
		tour = open(filename).read().split()[1:]
		tour = list(map(int, tour))
		if tour[-1] == 0: tour.pop()
		tour = tour+[0]
	
	#print(tour)
	return tour

def score_tour(tour,citiesT):
	df = citiesT.reindex(tour).reset_index()
	if len(citiesT)<50:
		print(df)
	primes = list(sympy.primerange(0, len(cities)))
	df['prime'] = df.CityId.isin(primes).astype(int)
	df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
	df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
	return df.dist.sum() + df.penalty.sum()

def score_path(dfPath):

	primes = list(sympy.primerange(0, len(cities)))
	dfPath['prime'] = dfPath.CityId.isin(primes).astype(int)
	dfPath['dist'] = np.hypot(dfPath.X - dfPath.X.shift(-1), dfPath.Y - dfPath.Y.shift(-1))
	dfPath['penalty'] = dfPath['dist'][9::10] * (1 - dfPath['prime'][9::10]) * 0.1
	return dfPath.dist.sum() + dfPath.penalty.sum()
	
def write_submission(tour, filename):
	assert set(tour) == set(range(len(tour)))
	pd.DataFrame({'Path': list(tour) + [0]}).to_csv(filename, index=False)


def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
	with open(filename, 'w') as f:
		f.write('NAME : %s\n' % name)
		f.write('COMMENT : %s\n' % name)
		f.write('TYPE : TSP\n')
		f.write('DIMENSION : %d\n' % len(cities))
		f.write('EDGE_WEIGHT_TYPE : EXPLICIT\n')
		f.write('EDGE_WEIGHT_FORMAT : FULL_MATRIX\n')
		f.write('EDGE_WEIGHT_SECTION\n')
		idx1 = 0
		for row in cities.itertuples():
			mstr = ''
			idx2 = 0
			for row1 in cities.itertuples():
				if idx1 == 0 and idx2 == len(cities)-1:
					mstr += '0 '
				elif idx1 == len(cities)-1 and idx2 == 0:
					mstr += '0 '
				else:
					mstr += str(int(((row.X-row1.X)**2+(row.Y-row1.Y)**2)**.5))+' '
				idx2+=1
			f.write(mstr+'\n')
			idx1+=1
		f.write('EOF\n')


def improveSome(nl,nm):
	tour = read_tour('linkernKernel.tour',nl,nm)
	savedScore = score_tour(tour,cities)
	#print(savedScore)
	cSome = []
	cSome.append(['CityId','X','Y'])
	dfSome = cities.reindex(tour).reset_index()
	dfSome.to_csv('citiesSome.csv')
	getSome = readcsv('citiesSome.csv')
	#print(getSome[1][1],getSome[-1][1])
	savedVals = [getSome[1][1],getSome[-1][1],savedScore]
	for i in getSome[1:]:
		cSome.append([i[1],i[2],i[3]])
	writecsv(cSome,'citiesSome.csv')

	citiesSome = pd.read_csv('citiesSome.csv', index_col=['CityId'])
	citiesSome1k=citiesSome*1000
	#print(citiesSome)
	write_tsp(citiesSome1k, 'citiesSome.tsp')
	for seedI in range(5,10):
		subprocess.run("./linkern -s "+str(random.randint(0,50))+" -S linkernSome.tour -R 1000000000 -t 10 ./citiesSome.tsp >linkernSome.log", shell=True, check=True)
		if nl >0:
			tour = read_tour('linkernSome.tour',nl,0)
		else:
			tour = read_tour('linkernSome.tour',nl,nm)
		#print(tour[1],tour[2])
		#print(cSome[tour[1]+1],cSome[tour[2]+1])
		pathSome = []
		pathSome.append(['CityId','X','Y'])
		for i in tour:
			pathSome.append([cSome[i+1][0],cSome[i+1][1],cSome[i+1][2]])
		writecsv(pathSome,'citiesSomePath.csv')
		dfPath = pd.read_csv('citiesSomePath.csv')
		#print(dfPath)
		if int(pathSome[1][0])==int(savedVals[0]) and int(pathSome[-1][0])==int(savedVals[1]):
			if score_path(dfPath)<savedVals[2]-.01:
				print(savedVals[2]-score_path(dfPath))
				savedVals[2]=score_path(dfPath)
				for iiidx,iii in enumerate(pathSome[1:]):
					newTour[iiidx+nl]=iii[0]
		#print(score_tour(tour,citiesSome))
	with open('improvedSubmission.csv', 'w') as f:
		f.write('Path\n')
		for iii in newTour:
			f.write(str(iii)+'\n')
	
	
cities = pd.read_csv('cities.csv', index_col=['CityId'])
newTour = []
for i in range(0,1):
	#subprocess.run("./linkern -s "+str(10+i)+" -S linkern.tour -R 1000000000 -t 1200 ./cities1k.tsp >linkern.log", shell=True, check=True)
	tour = read_tour('linkernKernel.tour',0,0)
	for ii in tour:
		newTour.append(ii)
	print(score_tour(tour,cities))
	#for ii in range(0,20):
	#	improveSome(8000*ii,8000*(ii+1))
	#for ii in range(0,45):
	#	improveSome(4000*ii,4000*(ii+1))
	for ii in range(0,95):
		improveSome(2000*ii,2000*(ii+1))
	#for ii in range(0,190):
	#	improveSome(1000*ii,1000*(ii+1))
	
	



