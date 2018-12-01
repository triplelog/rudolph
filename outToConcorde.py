import time
import sys
import random
import csv
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
	

#citiesCSV = readcsv('cities.csv')
cities = pd.read_csv('cities.csv', index_col=['CityId'])
cities1k = cities * 1000
def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index+1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities1k, 'cities1k.tsp')






