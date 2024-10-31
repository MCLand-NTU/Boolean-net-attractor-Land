# Name: Ket Hing Chong
# Date: 30 Oct 2024
# Affiliation: Nanyang Technological University

# Python program to print all paths from a source to destination.
  
from collections import defaultdict


# This class represents a directed graph 
# using adjacency list representation
class GraphObj:
  
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices 
         
        # default dictionary to store graph
        self.graph = defaultdict(list) 
        

  
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        

  
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path):

        global AllTrajectory

 
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
 
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            print('path=',path)
            AllTrajectory[d].append(path[:]) # Instead, append a copy of the list
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i]== False:
                    self.printAllPathsUtil(i, d, visited, path)
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False
        
  
    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
 
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
 
        # Create an array to store paths
        path = []
        
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)
        
          
  


# Using igraph and pajek input file

from igraph import *

graph1=Graph.Read_Pajek("cellcycle.net")

#graph1.vs["name"]=["0","1","2","3","4","5","6","7","8","9"]
#graph1.vs["label"]=graph1.vs["name"]

graph1.es["edge_arrow_size"]=[15]

summary(graph1)
layout=graph1.layout("kk")

#plot(graph1,layout=layout, bbox=(800,800), margin=100)

list1=graph1.get_edgelist()
print('list1 =', list1)

listNode=[]
listNodeIn=[]
listNodeOut=[]


# create a class DictList to enable duplicate keys in dictionary
class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)


# create an edges dictionary variabale
edges = Dictlist()

for edge in list1:
    # add item to edges with key=edge[0] and value=edge[1] because edge is in (x,y)
    edges[edge[0]]= edge[1]
    
print("edges=",edges)

# get the number of vertices in a graph created by Python igraph 
total_vertices=graph1.vcount()
print('total_vertices=', total_vertices)
total_edges=graph1.ecount()
print('total_edges=', total_edges)

#Create a Graph object and name it g2
g2=GraphObj(total_vertices)

for edge in graph1.es:
    source_vertex_id = edge.source
    target_vertex_id = edge.target
    g2.addEdge(source_vertex_id, target_vertex_id)

AllTrajectory = {}
for i in range(g2.V):
    AllTrajectory[i]=[]



# This code is contributed by Neelam Yadav

#print('In Graph class, Trajectory=',Trajectory)



countn = [0]*g2.V
print('countn =', countn)
print("")

for i in range(g2.V):
    for j in range(g2.V):
        s = i ; d = j
        #print ("Following are all different paths from % d to % d :" %(s, d))
        g2.printAllPaths(s, d)

print('AllTrajectory=',AllTrajectory)

print("")
for i in range(g2.V):
    print("AllTrajectory["+str(i)+"]=", AllTrajectory[i])

print("")
for i in range(g2.V):
    countn[i] = len(AllTrajectory[i])
    
print('countn =', countn)

count=countn
# calculate probability distribution 
nj=[0]*graph1.vcount()

total=sum(countn)
print('total=', total)
probability_n=[x / total for x in countn]

#print('probability_n =', probability_n)

nj=countn
pj = [ round(elem, 4) for elem in probability_n ]
#print('my_rounded_Probn =', pj )
print('pj =', pj )

import numpy as np

uj=[0]*graph1.vcount()
for i in range(len(count)):
    uj[i]=-np.log(pj[i])

uj = [ round(elem,3) for elem in uj]

print()
print('uj:',uj)

# code for finding Attractors and types of attractor
attractors = {}
steadyState = 'Steady State'
limitCycle = 'Limit Cycle'
normalNode = 'Normal Node'

def IsAttractor(preState, edges):
    print('preState',preState)
    if preState == edges[preState]:
        return 1
    trajectory = []
    tempState = preState
    while True:
        trajectory.append(tempState)
        
        nextState = edges[tempState]
        if nextState in trajectory:
            break
        tempState = nextState
    print('trajectory', trajectory)
    if trajectory.index(nextState) == 0:
        return len(trajectory)
    else:
        return 0


# plotting the Boolean attractor's landscape
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,7))
ax=fig.gca(projection='3d')

# plotting the nodes using scatter plot
# Get the x and y coordinates from the layout nodes position
a=np.zeros(shape=(len(layout),2))

for i in range(len(layout)):
    a[i]=layout[i]

# use transpose to get the x and y coordinates    
b=np.transpose(a)
# b[0]--> x
# b[1]--> y

n=len(count)
for c, m in [('r','o')]:
    xs=b[0]  # np.random.rand(len(count))
    ys=b[1]  # np.random.rand(len(count))
    zs=uj
    ax.scatter(xs,ys,zs,color='blue',marker=m)


for index in range(len(list1)):
    x=list1[index]
    listNodeIn.append(x[1])
    listNodeOut.append(x[0])

print('   listNodeIn=',listNodeIn)
print('   listNodeOut=',listNodeOut)
print()

# Plotting arrows directed graph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

#ax = fig.gca(projection='3d')
#ax.set_aspect("equal")

#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

print('Draw directed arrows with the function Arrow3D:')
for j in range(len(listNodeOut)):
    #print('j:',j,'listNodeOut[j]:',listNodeOut[j],'listNodeIn[j]:',listNodeIn[j])
    #a = Arrow3D([xs[listNodeOut[j]], xs[listNodeIn[j]]],[ys[listNodeOut[j]], ys[listNodeIn[j]]],[uj[listNodeOut[j]], uj[listNodeIn[j]]], mutation_scale=20, lw=0.5, arrowstyle="-|>", color="red") arrow head sharp
    a = Arrow3D([xs[listNodeOut[j]], xs[listNodeIn[j]]],[ys[listNodeOut[j]], ys[listNodeIn[j]]],[uj[listNodeOut[j]], uj[listNodeIn[j]]], mutation_scale=10, lw=0.3, arrowstyle="->", color="red")
    #a = Arrow3D([xs[listNodeOut[j]], xs[listNodeIn[j]]],[ys[listNodeOut[j]], ys[listNodeIn[j]]],[uj[listNodeOut[j]], uj[listNodeIn[j]]], mutation_scale=20, lw=1, arrowstyle="->", color="red") arrow with two lines
    #b = Arrow3D([0,2],[0,2],[0,2], mutation_scale=20, lw=2, arrowstyle="-|>", color="r")
    #ax.add_artist(b)
    ax.add_artist(a)
print('number of directed arrows=',len(listNodeOut))
print()
# Draw selfloop arc with arrows
selfLoop_index=[]
for i in range(0,len(listNodeIn)):
    if listNodeIn[i]==listNodeOut[i]:
        print('self loop index',i)
        selfLoop_index.append(i)

for i in selfLoop_index:        
    ax.plot([xs[listNodeIn[i]]-0.05],[ys[listNodeIn[i]]-0.05],[uj[listNodeIn[i]]-0.05],marker=r'$\circlearrowleft$', ms=10, color='red')

# Finding attractor index and label the attractor with blue color node
noOutgoing_index=[]
for i in range(graph1.vcount()):
    if not (i in listNodeOut):
        print('no outgoing index:',i)
        noOutgoing_index.append(i)

attractor_index=[]
for i in range(graph1.vcount()):
    if (i in noOutgoing_index or i in selfLoop_index):
        print('attractor node index:',i)
        attractor_index.append(i)

print('attractor_index:',attractor_index)

  # Superimpose with another scatter plot
  # for drawing attractor node
  # python node index=25
  # one stable point attractor
for i in attractor_index:
    ax.scatter(
       xs[i],ys[i],zs[i],
       color='yellow',
       marker='o',
       s=100
    )

## finding cyclic attractor
#cyclic_attractor_index=[]
#for i in range(graph1.vcount()):
    #cyclicA=ReachedAttractor(i,edges)
    #print(' cyclicA:',cyclicA)
    ##print(' trajectory:',trajectory)
    
    
#print('cyclic_attractor:',cyclic_attractor_index)

  ## Superimpose with another scatter plot
  ## for drawing cyclic attractor
#for i in cyclic_attractor_index:
    #ax.scatter(
    #xs[i],ys[i],zs[i],
    #color='pink',
    #marker='o',
    #s=100
    #)

    # https://stackoverflow.com/questions/40833612/find-all-cycles-in-a-graph-implementation
    
def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph.get(state, []):
        #for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))


graph = { 1: [2, 3, 5], 2: [1], 3: [1], 4: [2], 5: [2] }
cycles = [[node]+path  for node in graph for path in dfs(graph, node, node)]

print(cycles)
# output: [[1, 5, 2, 1], [1, 3, 1], [1, 2, 1], [2, 1, 5, 2], [2, 1, 2], [3, 1, 3], [5, 2, 1, 5]]

#edges= {0: [1], 1: [2, 6], 2: [0], 3: [1, 5], 4: [3, 5], 5:[], 6: [3, 7], 7: [8], 8: [6], 9: [6, 7, 8]}

#I tried graph = {2: [4, 1], 3: [2], 1: [4, 3]} but it always come out KeyError: 4 – 
#nosense
# Dec 17, 2019 at 3:36 
#1
#Your graph doesn't describe the complete graph (i.e. it misses a definition for node 4). You can either add 4: [] to your graph definition, or you can replace for next_state in graph[state]: with for next_state in graph.get(state, []):. – 
#AChampion
# Dec 17, 2019 at 22:53

graph2 = edges
allCycles = [[node]+path  for node in graph2 for path in dfs(graph2, node, node)]
print('All cycles in the directed graph:', allCycles)

# finding cyclic attractor
cyclic_attractor_index=[]
for i in range(len(allCycles)):
    #print(allCycles[i])
    for j in allCycles[i]:
        #print(j)
        if j not in cyclic_attractor_index:
            #print(j)
            cyclic_attractor_index.append(j)
print(' cyclic_attractor_index:',cyclic_attractor_index)

 #Superimpose with another scatter plot
 #for drawing cyclic attractor
for i in cyclic_attractor_index:
    ax.scatter(
    xs[i],ys[i],zs[i],
    color='pink',
    marker='o',
    s=100
    )    


# Add annotation of node number to the 3D graph
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)



xyzn = list(zip(xs, ys, zs))
#segments = [(xyzn[s], xyzn[t]) for s, t in edges]    

# add vertices annotation.
#for j, xyz_ in enumerate(xyzn): 
    #annotate3D(ax, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3,3),
               #textcoords='offset points', ha='right',va='bottom') 

    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U=-ln(P)')

#ax.view_init(elev=127, azim=28)

plt.savefig('FigCellcycleLargerFig.eps', format='eps')

