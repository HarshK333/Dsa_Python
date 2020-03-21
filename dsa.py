### Graph ###
dictGraph={}
queue=[]
min_dist={}
max_n=10**9
dictUnion={}
def init_dictGraph(vertices):
    for i in vertices:
        dictGraph[i]=[]

def addEdge(u,v):
    dictGraph[u].append(v)

def addWeightedEdge(u,v,wt):
    dictGraph[u].append([v,wt])

def multi_dem_list_to_single_list(a,rows,cols):
    b=[]
    for i in range(rows):
        for j in range(cols):
            b.append(a[i][j])
    #12 is 1*cols+j
    return b  

def DFS(v,visited):
    visited[v]=True
    print(v)
    while dictGraph[v]!=[]:
        x=dictGraph.pop()
        if visited[x]==False:
            DFS(x,visited)


def BFS(queue,visited):
    while queue!=[]:
        x=queue.pop(0)
        if visited[x]==False:
            visited[x]=True
            print(x)
            for v in dictGraph[x]:
                if visited[v]==False:
                    queue.append(v)

def init_djikstra(vertices,src,dest):
    for i in vertices:
        min_dist[i]=max_n
    min_dist[src]=0
    visited=[False]*len(vertices)
    djikstra(visited,src,dest)


def djikstra(visited,src,dest):
    visited[src]=True
    for i in dictGraph[src]:
        if min_dist[i[0]]>min_dist[src]+i[1]:
            min_dist[i[0]]=min_dist[src]+i[1]
    mini=10**9
    mini_i=-1
    for i in min_dist:
        if mini>min_dist[i] and visited[i]==False:
            mini=min_dist[i]
            mini_i=i
    if mini_i!=-1 and visited[dest]==False:
        djikstra(visited,mini_i,dest)


def initial_union_nod(vertices):
    for i in vertices:
        dictUnion[i]=-1

def union(a,b):
    p1=find(a)
    p2=find(b)
    rank1=-1*dictUnion[p1]
    rank2=-1*dictUnion[p2]
    if p1!=p2:
        if rank1>rank2:
            dictUnion[p2]=dictUnion[p1]
        else:
            dictUnion[p1]=dictUnion[p2]

def find(a):
    x=dictUnion[a]
    p=a
    while x>=0:
        p=x
        x=dictUnion[x]
    dictUnion[a]=p
    return p
    

### kth largest element ###
#from dsapy.heaps import MinHeap
from heapq import heappush, heappop, heapify 
#O(n+klogn)
def kth_smallest(k,li):
	l=[]
	heapify(li)
	for i in range(k):
		l.append(heappop(li))
	return (l[k-1])
def kth_largest(k,li):
	a=max(li)
	for i in range(len(li)):
		li[i]=a-li[i]
	l=[]
	heapify(li)
	for i in range(k):
		l.append(heappop(li))
	return (a-l[k-1])

li=[6,5,4,2,3]
print(li)
k=int(input("kth largest elemnt no:"))
print(kth_largest(k,li))
k=int(input("kth smallest elemnt no:"))
print(kth_smallest(k,li))

### segment trees ###
"""
    The idea here is to build a segment tree. Each node stores the left and right
    endpoint of an interval and the sum of that interval. All of the leaves will store
    elements of the array and each internal node will store sum of leaves under it.
    Creating the tree takes O(n) time. Query and updates are both O(log n).
"""

#Segment tree node
class Node:
    def __init__(self,start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None
        

class NumArray:
    def __init__(self, nums):

        """
        initialize your data structure here.
        :type nums: List[int]
        """
        #helper function to create the tree from input array
        def createTree(nums, l, r):
            
            #base case
            if l > r:
                return None
                
            #leaf node
            if l == r:
                n = Node(l, r)
                n.total = nums[l]
                return n
            
            mid = (l + r) // 2
            
            root = Node(l, r)
            
            #recursively build the Segment tree
            root.left = createTree(nums, l, mid)
            root.right = createTree(nums, mid+1, r)
            
            #Total stores the sum of all leaves under root
            #i.e. those elements lying between (start, end)
            root.total = root.left.total + root.right.total
                
            return root
        
        self.root = createTree(nums, 0, len(nums)-1)
            
    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        #Helper function to update a value
        def updateVal(root, i, val):
            
            #Base case. The actual value will be updated in a leaf.
            #The total is then propogated upwards
            if root.start == root.end:
                root.total = val
                return val
        
            mid = (root.start + root.end) // 2
            
            #If the index is less than the mid, that leaf must be in the left subtree
            if i <= mid:
                updateVal(root.left, i, val)
                
            #Otherwise, the right subtree
            else:
                updateVal(root.right, i, val)
            
            #Propogate the changes after recursive call returns
            root.total = root.left.total + root.right.total
            
            return root.total
        
        return updateVal(self.root, i, val)

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        #Helper function to calculate range sum
        def rangeSum(root, i, j):
            
            #If the range exactly matches the root, we already have the sum
            if root.start == i and root.end == j:
                return root.total
            
            mid = (root.start + root.end) // 2
            
            #If end of the range is less than the mid, the entire interval lies
            #in the left subtree
            if j <= mid:
                return rangeSum(root.left, i, j)
            
            #If start of the interval is greater than mid, the entire inteval lies
            #in the right subtree
            elif i >= mid + 1:
                return rangeSum(root.right, i, j)
            
            #Otherwise, the interval is split. So we calculate the sum recursively,
            #by splitting the interval
            else:
                return rangeSum(root.left, i, mid) + rangeSum(root.right, mid+1, j)
        
        return rangeSum(self.root, i, j)
                


# Your NumArray object will be instantiated and called as such:
nums=[5,4,3,2,1,6,7,8,9]
numArray = NumArray(nums)
print(numArray.sumRange(0, 1))
print(numArray.update(1, 10))
print(numArray.sumRange(1, 2))
print(numArray.sumRange(0, 1))
print(numArray.update(1, 10))
print(numArray.sumRange(1, 2))


### Topological_sort ###
#Python program to print topological sorting of a DAG 
from collections import defaultdict 

#Class to represent a graph 
class Graph: 
	def __init__(self,vertices): 
		self.graph = defaultdict(list) #dictionary containing adjacency List 
		self.V = vertices #No. of vertices 

	# function to add an edge to graph 
	def addEdge(self,u,v): 
		self.graph[u].append(v) 

	# A recursive function used by topologicalSort 
	def topologicalSortUtil(self,v,visited,stack): 

		# Mark the current node as visited. 
		visited[v] = True

		# Recur for all the vertices adjacent to this vertex 
		for i in self.graph[v]: 
			if visited[i] == False: 
				self.topologicalSortUtil(i,visited,stack) 

		# Push current vertex to stack which stores result 
		stack.insert(0,v) 

	# The function to do Topological Sort. It uses recursive 
	# topologicalSortUtil() 
	def topologicalSort(self): 
		# Mark all the vertices as not visited 
		visited = [False]*self.V 
		stack =[] 

		# Call the recursive helper function to store Topological 
		# Sort starting from all vertices one by one 
		for i in range(self.V): 
			if visited[i] == False: 
				self.topologicalSortUtil(i,visited,stack) 

		# Print contents of the stack 
		print(stack) 

g= Graph(6) 
g.addEdge(5, 2); 
g.addEdge(5, 0); 
g.addEdge(4, 0); 
g.addEdge(4, 1); 
g.addEdge(2, 3); 
g.addEdge(3, 1); 

print("Following is a Topological Sort of the given graph")
g.topologicalSort() 
#This code is contributed by Neelam Yadav 


### sort_nearly_sorted_array ###
#k is the maximum distance of any element from its position in sorted array
from heapq import heappush, heappop, heapify 
#O(k) + O((n-k)*logK)
def sort_array(l,k):
	a=l[:k+1]
	heapify(a)
	l1=[]
	for i in range(k+1,len(l)):
		x=l[i]
		l1.append(heappop(a))
		heappush(a,x)
	while a:
		l1.append(heappop(a))
	return l1


# Driver Code 
k = 3
arr = [2, 6, 3, 12, 56, 8] 
l1=sort_array(arr, k)   
print('Following is sorted array') 
print(l1) 



### Segment treesrange minimum ###
"""
    The idea here is to build a segment tree. Each node stores the left and right
    endpoint of an interval and the sum of that interval. All of the leaves will store
    elements of the array and each internal node will store sum of leaves under it.
    Creating the tree takes O(n) time. Query and updates are both O(log n).
"""

#Segment tree node
import sys
class Node:
    def __init__(self,start, end):
        self.start = start
        self.end = end
        self.mini =10**9
        self.left = None
        self.right = None
        

class NumArray:
    def __init__(self, nums):

        """
        initialize your data structure here.
        :type nums: List[int]
        """
        #helper function to create the tree from input array
        def createTree(nums, l, r):
            
            #base case
            if l > r:
                return None
                
            #leaf node
            if l == r:
                n = Node(l, r)
                n.total = nums[l]
                return n
            
            mid = (l + r) // 2
            
            root = Node(l, r)
            
            #recursively build the Segment tree
            root.left = createTree(nums, l, mid)
            root.right = createTree(nums, mid+1, r)
            
            #Total stores the sum of all leaves under root
            #i.e. those elements lying between (start, end)
            root.total = min(root.left.total,root.right.total)
                
            return root
        
        self.root = createTree(nums, 0, len(nums)-1)
            
    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        #Helper function to update a value
        def updateVal(root, i, val):
            
            #Base case. The actual value will be updated in a leaf.
            #The total is then propogated upwards
            if root.start == root.end:
                root.total = val
                return val
        
            mid = (root.start + root.end) // 2
            
            #If the index is less than the mid, that leaf must be in the left subtree
            if i <= mid:
                updateVal(root.left, i, val)
                
            #Otherwise, the right subtree
            else:
                updateVal(root.right, i, val)
            
            #Propogate the changes after recursive call returns
            root.total = min(root.left.total,root.right.total)
            
            return root.total
        
        return updateVal(self.root, i, val)

    def minRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        #Helper function to calculate range sum
        def rangemin(root, i, j):
            
            #If the range exactly matches the root, we already have the sum
            if root.start == i and root.end == j:
                return root.total
            
            mid = (root.start + root.end) // 2
            
            #If end of the range is less than the mid, the entire interval lies
            #in the left subtree
            if j <= mid:
                return rangemin(root.left, i, j)
            
            #If start of the interval is greater than mid, the entire inteval lies
            #in the right subtree
            elif i >= mid + 1:
                return rangemin(root.right, i, j)
            
            #Otherwise, the interval is split. So we calculate the sum recursively,
            #by splitting the interval
            else:
                return min(rangemin(root.left, i, mid),rangemin(root.right, mid+1, j))
        
        return rangemin(self.root, i, j)
                


# Your NumArray object will be instantiated and called as such:
nums=[5,4,3,2,1,6,7,8,9]
numArray = NumArray(nums)
print(numArray.minRange(0, 1))
print(numArray.update(1, 10))
print(numArray.minRange(1, 2))
print(numArray.minRange(0, 1))
print(numArray.update(1, 10))
print(numArray.minRange(1, 2))


### dijikstra min_heap ###
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


### fast_modulo ###
def fast_exp(base,exp):
	res=1
	MOD=10**9+7
	while(exp>0):
		if(exp%2==1):
			res=(res*base)%MOD
		base=(base*base)%MOD
		exp=exp//2
	return res%MOD

#driver code 
base=int(input("base:"))
expo=int(input("expo:"))
print(fast_exp(base, expo))



### Gcd Or LCM
import math
gc=math.gcd(x,y)
lc=x*y//gcd(x,y)
def gcd(a,b): 
    if a == 0: 
        return b 
    return gcd(b % a, a) 
  
# Function to return LCM of two numbers 
def lcm(a,b): 
    return ((a*b) // gcd(a,b))



## binary_search ##

def binary_search(a,v):
    lo = 1
    hi = len(a)
    while( lo <= hi ):
        mid = (lo+hi)//2
        if a[mid] == v:
            return mid
        if a[mid] > v:
            hi = mid-1
        else:
            lo = mid+1
    return -1

#driver code
arr=[1,4,7,9]
ele=int(input("ele:"))
print(binary_search(arr,ele)) 


### primeNumbersSieve ###
import math 
def primeNumbers(n):
	a=[]
	bol=[]
	for i in range(2,n+1):
		a.append(i)
		bol.append(1)
	b=math.sqrt(n)
	for i in a:
		if i>b:
			break
		for j in range(i*i,n+1,i):
			if bol[j-2]==1:
				a.remove(j)
				bol[j-2]=0
	return a

#driver code
n=int(input("find prime numbers in range of :"))
print(primeNumbers(n))



### primefactors ###
import math 
  
# A function to print all prime factors of  
# a given number n 
#Time complexity: O(sqrt(n))
def primeFactors(n): 
      
    # Print the number of two's that divide n 
    l=[]
    while n % 2 == 0: 
        l.append(2)
        n = n / 2
          
    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(math.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
        while n % i== 0: 
            l.append(i)
            n = n / i 

              
    # Condition if n is a prime 
    # number greater than 2 
    if n > 2: 
        l.append(int(n)) 
    return l
          
# Driver Program to test above function 
  
n = int(input())
l=primeFactors(n) 
print(l)

### kruskal_min_spanning ###

# Python program for Kruskal's algorithm to find 
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph 
#O(ElogE) or O(ElogV)
from collections import defaultdict 

#Class to represent a graph 
class Graph: 

	def __init__(self,vertices): 
		self.V= vertices #No. of vertices 
		self.graph = [] # default dictionary 
								# to store graph 
		

	# function to add an edge to graph 
	def addEdge(self,u,v,w): 
		self.graph.append([u,v,w]) 

	# A utility function to find set of an element i 
	# (uses path compression technique) 
	def find(self, parent, i): 
		if parent[i] == i: 
			return i 
		return self.find(parent, parent[i]) 

	# A function that does union of two sets of x and y 
	# (uses union by rank) 
	def union(self, parent, rank, x, y): 
		xroot = self.find(parent, x) 
		yroot = self.find(parent, y) 

		# Attach smaller rank tree under root of 
		# high rank tree (Union by Rank) 
		if rank[xroot] < rank[yroot]: 
			parent[xroot] = yroot 
		elif rank[xroot] > rank[yroot]: 
			parent[yroot] = xroot 

		# If ranks are same, then make one as root 
		# and increment its rank by one 
		else : 
			parent[yroot] = xroot 
			rank[xroot] += 1

	# The main function to construct MST using Kruskal's 
		# algorithm 
	def KruskalMST(self): 

		result =[] #This will store the resultant MST 

		i = 0 # An index variable, used for sorted edges 
		e = 0 # An index variable, used for result[] 

			# Step 1: Sort all the edges in non-decreasing 
				# order of their 
				# weight. If we are not allowed to change the 
				# given graph, we can create a copy of graph 
		self.graph = sorted(self.graph,key=lambda item: item[2]) 

		parent = [] ; rank = [] 

		# Create V subsets with single elements 
		for node in range(self.V): 
			parent.append(node) 
			rank.append(0) 
	
		# Number of edges to be taken is equal to V-1 
		while e < self.V -1 : 

			# Step 2: Pick the smallest edge and increment 
					# the index for next iteration 
			u,v,w = self.graph[i] 
			i = i + 1
			x = self.find(parent, u) 
			y = self.find(parent ,v) 

			# If including this edge does't cause cycle, 
						# include it in result and increment the index 
						# of result for next edge 
			if x != y: 
				e = e + 1	
				result.append([u,v,w]) 
				self.union(parent, rank, x, y)			 
			# Else discard the edge 

		# print the contents of result[] to display the built MST 
		print "Following are the edges in the constructed MST"
		for u,v,weight in result: 
			#print str(u) + " -- " + str(v) + " == " + str(weight) 
			print ("%d -- %d == %d" % (u,v,weight)) 

# Driver code 
g = Graph(4) 
g.addEdge(0, 1, 10) 
g.addEdge(0, 2, 6) 
g.addEdge(0, 3, 5) 
g.addEdge(1, 3, 15) 
g.addEdge(2, 3, 4) 

g.KruskalMST() 

#This code is contributed by Neelam Yadav 


### dfs_connectdcomp ###

# Python program to print connected  
# components in an undirected graph 
class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited): 

        # Mark the current vertex as visited 
        visited[v] = True

        # Store the vertex to list 
        temp.append(v) 

        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 

    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 

    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc 
  

# Create a graph given in the above diagram 
# 5 vertices numbered from 0 to 4 
g = Graph(5); 
g.addEdge(1, 0) 
g.addEdge(2, 3) 
g.addEdge(3, 4) 
cc = g.connectedComponents() 
print("Following are connected components") 
print(cc) 


### dfs_transitive_closure ###
# Python program to print transitive closure of a graph 
from collections import defaultdict 

# This class represents a directed graph using adjacency 
# list representation 
class Graph: 

    def __init__(self,vertices): 
        # No. of vertices 
        self.V= vertices 

        # default dictionary to store graph 
        self.graph= defaultdict(list) 

        # To store transitive closure 
        self.tc = [[0 for j in range(self.V)] for i in range(self.V)] 

    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 

    # A recursive DFS traversal function that finds 
    # all reachable vertices for s 
    def DFSUtil(self,s,v): 

        # Mark reachability from s to v as true. 
        self.tc[s][v] = 1

        # Find all the vertices reachable through v 
        for i in self.graph[v]: 
            if self.tc[s][i]==0: 
                self.DFSUtil(s,i) 

    # The function to find transitive closure. It uses 
    # recursive DFSUtil() 
    def transitiveClosure(self): 

        # Call the recursive helper function to print DFS 
        # traversal starting from all vertices one by one 
        for i in range(self.V): 
            self.DFSUtil(i, i) 
        print(self.tc) 

# Create a graph given in the above diagram 
g = Graph(4) 
g.addEdge(0, 1) 
g.addEdge(0, 2) 
g.addEdge(1, 2) 
g.addEdge(2, 0) 
g.addEdge(2, 3) 
g.addEdge(3, 3) 

print("Transitive closure matrix is")
g.transitiveClosure(); 

# This is enddddddddddddd