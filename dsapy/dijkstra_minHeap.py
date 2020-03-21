# A Python program for Dijkstra's shortest 
# path algorithm for adjacency 
# list representation of graph 

from collections import defaultdict 
import sys 


from heapq import heappush, heappop, heapify 

# heappop - pop and return the smallest element from heap 
# heappush - push the value item onto the heap, maintaining 
#			 heap invarient 
# heapify - transform list into heap, in place, in linear time 

# A class for Min Heap 
class MinHeap: 
	
	# Constructor to initialize a heap 
	def __init__(self,l):
		heapify_l(l)
		self.heap = list(l)

	def heapify_l(self,l):
	    size=len(l)
	    for root in range((size//2)-1,-1,-1):
	        root_val = l[root]             # save root value
	        child = 2*root+1
	        while(child<size):
	            if child<size-1 and l[child]>l[child+1]:
	                child+=1
	            if root_val<=l[child]:     # compare against saved root value
	                break
	            l[(child-1)//2]=l[child]   # find child's parent's index correctly
	            child=2*child+1
	        l[(child-1)//2]=root_val       # here too, and assign saved root value
	    return l

# 	def parent(self, i): 
# 		return (i-1)/2
	
# 	# Inserts a new key 'k' 
# 	def insertKey(self, k): 
# 		heappush(self.heap, k)		 

# 	# Decrease value of key at index 'i' to new_val 
# 	# It is assumed that new_val is smaller than heap[i] 
# 	def decreaseKey(self, i, new_val): 
# 		self.heap[i] = new_val 
# 		while(i != 0 and self.heap[self.parent(i)] > self.heap[i]): 
# 			# Swap heap[i] with heap[parent(i)] 
# 			self.heap[i] , self.heap[self.parent(i)] = ( 
# 			self.heap[self.parent(i)], self.heap[i]) 
			
# 	# Method to remove minium element from min heap 
# 	def extractMin(self): 
# 		return heappop(self.heap) 

# 	# This functon deletes key at index i. It first reduces 
# 	# value to minus infinite and then calls extractMin() 
# 	def deleteKey(self, i): 
# 		self.decreaseKey(i, float("-inf")) 
# 		self.extractMin() 

# 	# Get the minimum element from the heap 
# 	def getMin(self): 
# 		return self.heap[0] 

# 	def getHeap(self):
# 		return self.heap


# class Graph(): 

# 	def __init__(self, vertices): 
# 		self.V = vertices 
# 		self.graph = [[]]*vertices
# 		self.l=[1 for v in vertices]

# 	# Adds an edge to an undirected graph 
# 	def addEdge(self, src, dest, weight): 

# 		# Add an edge from src to dest. A new node 
# 		# is added to the adjacency list of src. The 
# 		# node is added at the beginning. The first 
# 		# element of the node has the destination 
# 		# and the second elements has the weige[0]ht 
# 		newNode = [dest, weight] 
# 		self.graph[src].append(newNode) 

# 		# Since graph is undirected, add an edge 
# 		# from dest to src also 
# 		newNode = [src, weight] 
# 		self.graph[dest].append(newNode) 

# 	# The main function that calulates distances 
# 	# of shortest paths from src to all vertices. 
# 	# It is a O(ELogV) function 
# 	def dijkstra(self, src): 

# 		V = self.V # Get the number of vertices in graph 
# 		dist = [] # dist values used to pick minimum 
# 					# weight edge in cut 
# 		l=self.l
# 		# Initialize min heap with all vertices. 
# 		# dist value of all vertices 
# 		for v in range(V): 
# 			dist.append([sys.maxint,v]) 

# 		# Make dist value of src vertex as 0 so 
# 		# that it is extracted first \
# 		dist[src][0] = 0
# 		minHeap = MinHeap(dist) 
 

# 		# In the following loop, min heap contains all nodes 
# 		# whose shortest distance is not yet finalized. 
# 		while l: 

# 			# Extract the vertex with minimum distance value 
# 			newHeapNode= minHeap.extractMin()
# 			u = newHeapNode[1]
# 			l[u]=0

# 			# Traverse through all adjacent vertices of 
# 			# u (the extracted vertex) and update their 
# 			# distance values 
# 			for pCrawl in self.graph[u]: 

# 				v = pCrawl[0] 

# 				# If shortest distance to v is not finalized 
# 				# yet, and distance to v through u is less 
# 				# than its previously calculated distance 
# 				if l[v]==1 and pCrawl[1] + dist[u][0] < dist[v][0]: 
# 						dist[v][0] = pCrawl[1] + dist[u][0] 

# 						# update distance value 
# 						# in min heap also 
# 						minHeap.decreaseKey(dist[v]) 

# 		printArr(dist,V) 


# # Driver program to test the above functions 
# graph = Graph(9) 
# graph.addEdge(0, 1, 4) 
# graph.addEdge(0, 7, 8) 
# graph.addEdge(1, 2, 8) 
# graph.addEdge(1, 7, 11) 
# graph.addEdge(2, 3, 7) 
# graph.addEdge(2, 8, 2) 
# graph.addEdge(2, 5, 4) 
# graph.addEdge(3, 4, 9) 
# graph.addEdge(3, 5, 14) 
# graph.addEdge(4, 5, 10) 
# graph.addEdge(5, 6, 2) 
# graph.addEdge(6, 7, 1) 
# graph.addEdge(6, 8, 6) 
# graph.addEdge(7, 8, 7) 
# graph.dijkstra(0) 

# # This code is contributed by Divyanshu Mehta 

def heapify_l(l,ver):
    size=len(l)
    for root in range((size//2)-1,-1,-1):
        #print(root)
        root_val = l[root]
        child = 2*root+1
        while(child<size):
            if child<size-1 and l[child]>l[child+1]:
                child+=1
            if root_val<=l[child]:     # compare against saved root value
                break
            i=l[child][1]
            i1=l[(child-1)//2][1]
            ver[i],ver[i1]=ver[i1],ver[i]
            print("1ver["+str(i)+"]="+str(ver[i]))
            print("1ver["+str(i1)+"]="+str(ver[i1]))
            l[(child-1)//2],l[child]=l[child],l[(child-1)//2]   # find child's parent's index correctly
            child=2*child+1
        # i=root_val[1]
        # ver[i]=l[(child-1)//2][1]
        # print("2ver["+str(i)+"]="+str(l[(child-1)//2][1]))
        # l[(child-1)//2]=root_val       # here too, and assign saved root value
    print(ver)
    return l

l=[[100,0],[50,1],[50,2],[100,3],[75,4],[0,5]]
ver=[0,1,2,3,4,5]
heapify_l(l,ver)
print(list(l))