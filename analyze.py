import time
import sys
import random
import csv
import math
import json
import numpy as np
import matplotlib.pyplot as plt


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
        
import numpy as np
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
	
primeCSV = readcsv('primes.csv')
submissionCSV = readcsv('submission.csv')
#submissionCSV = [0, 48816, 40230, 75405, 153911, 22121, 38941, 167366, 177242, 47239, 137367, 161041, 37119, 138832, 25707, 117141, 161210, 182152, 100069]
citiesCSV = readcsv('cities.csv')
allPrimes = []
defaultRoute = []


for i in range(0,len(primeCSV)):
	for ii in range(0,10):
		allPrimes.append(int(primeCSV[i][ii]))

for i in range(1,len(submissionCSV)):
	if int(submissionCSV[i][0]) in allPrimes:
		defaultRoute.append([int(submissionCSV[i][0]),1,float(citiesCSV[int(submissionCSV[i][0])+1][1]),float(citiesCSV[int(submissionCSV[i][0])+1][2])])
	else:
		defaultRoute.append([int(submissionCSV[i][0]),0,float(citiesCSV[int(submissionCSV[i][0])+1][1]),float(citiesCSV[int(submissionCSV[i][0])+1][2])])
print(len(defaultRoute))
print(defaultRoute[len(defaultRoute)-10:])


def improve1000(idx,defaultRoute):
	arr = {'tprimes':0,'cprimes':0,'gprimes':0,'tdistance':0,'baddistance':0,'dsaved':0}
	x = []
	y = []
	colors = []
	area = []
	cities = []
	for i in range(idx,idx+60):
		d = ((defaultRoute[i][2]-defaultRoute[i+1][2])**2+(defaultRoute[i][3]-defaultRoute[i+1][3])**2)
		x.append(defaultRoute[i][2])
		y.append(defaultRoute[i][3])
		if i%10==9:
			colors.append(0)
		else:
			colors.append(1)
		area.append(5*(1+defaultRoute[i][1]))
		if defaultRoute[i][1]==1:
			arr['tprimes']+=1
			if i%10==9:
				arr['gprimes']+=1
			elif 2==3:
				arr['cprimes']+=1
				
		elif i%10==9:
			arr['baddistance']+=d
			if d>5000:
				print(d)
				print(defaultRoute[i-169:i+171])
			
		arr['tdistance']+=d
		if i>idx and i<idx+59:
			cities.append([defaultRoute[i][2],defaultRoute[i][3]])
	
	if 3==2:
		citiesM = np.matrix([[defaultRoute[idx][2],defaultRoute[idx][3]]]+cities+[[defaultRoute[idx+59][2],defaultRoute[idx+59][3]]])
		route = np.arange(citiesM.shape[0])
		print(str(path_distance(route,citiesM)))
		route = two_opt(citiesM,.00001)
		#new_cities_order = np.array([citiesM[route[i]] for i in range(len(route))])
		print(str(path_distance(route,citiesM)))
	
		random.shuffle(cities)
		citiesM = np.matrix([[defaultRoute[idx][2],defaultRoute[idx][3]]]+cities+[[defaultRoute[idx+59][2],defaultRoute[idx+59][3]]])
		route = two_opt(citiesM,.00001)
		#new_cities_order = np.array([citiesM[route[i]] for i in range(len(route))])
		print(str(path_distance(route,citiesM)))
	
		random.shuffle(cities)
		citiesM = np.matrix([[defaultRoute[idx][2],defaultRoute[idx][3]]]+cities+[[defaultRoute[idx+59][2],defaultRoute[idx+59][3]]])
		route = two_opt(citiesM,.00001)
		#new_cities_order = np.array([citiesM[route[i]] for i in range(len(route))])
		print(str(path_distance(route,citiesM)))
		print('')
		#print(citiesM)
	
		#plt.scatter(x, y, s=area, c=colors, alpha=0.75,zorder=2)
		#plt.plot(x,y,zorder=1)
		#plt.show()
		#if arr['baddistance']*1.0/arr['tdistance']>.075:
		#	print(arr['baddistance']*1.0/arr['tdistance'])
		#	print(defaultRoute[idx:idx+60])



distance = 0
maxd = 0
maxidx = -1
for idx,i in enumerate(defaultRoute):
	if idx < 197770-1:
		d = ((i[2]-defaultRoute[idx+1][2])**2+(i[3]-defaultRoute[idx+1][3])**2)
		if idx%10==9 and i[1]==0:
			distance += 1.1*d**.5
		else:
			distance += 1.0*d**.5
		if d > maxd:
			maxd = d
			maxidx = idx
	if idx==98:
		print(distance)
def all10distance(startI):
	
	backStart = -1
	dR = defaultRoute[startI+100:startI+1000]
	sX = defaultRoute[startI][2]
	sY = defaultRoute[startI][3]
	s2X = defaultRoute[startI+1][2]
	s2Y = defaultRoute[startI+1][3]
	dori = ((s2X-sX)**2+(s2Y-sY)**2)
	for idx,i in enumerate(dR):
		if idx < len(dR)-1:
			d = ((i[2]-sX)**2+(i[3]-sY)**2)
			doth = ((i[2]-dR[idx+1][2])**2+(i[3]-dR[idx+1][3])**2)
			dcross = -dori**.5-doth**.5+d**.5+((dR[idx+1][2]-s2X)**2+(dR[idx+1][3]-s2Y)**2)**.5
			
			if dcross<20:
				backStart = idx+startI+100
				break
	if backStart > -1:
		allDist = []
		for startII in range(0,10):
			distance = 0
			dR = defaultRoute[startI-10:backStart+11]
			for idx,i in enumerate(dR):
				if idx < len(dR)-1:
					d = ((i[2]-dR[idx+1][2])**2+(i[3]-dR[idx+1][3])**2)
					if idx%10==startII and i[1]==0:
						distance += 1.1*d**.5
					else:
						distance += 1.0*d**.5
			allDist.append(distance)
		if max(allDist)-min(allDist)>4.0:
			print(allDist[9]-min(allDist),allDist,backStart-startI, startI)
			print(dR)
		

#print(distance)
#for i in range(10,3001):
#	improve1000(i*60,defaultRoute)
for i in range(0,950):
	all10distance(i*200)
