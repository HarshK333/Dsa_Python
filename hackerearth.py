'''
# Sample code to perform I/O:

name = input()                  # Reading input from STDIN
print('Hi, %s.' % name)         # Writing output to STDOUT

# Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
'''

# Write your code here

# A utility function to find the vertex with 
# minimum distance value, from the set of vertices 
# not yet included in shortest path tree 
l=10**9
def minDistance(dist, sptSet,V): 

	# Initilaize minimum distance for next node 
	min = l

	# Search not nearest vertex not in the 
	# shortest path tree 
	for v in range(V): 
		if dist[v] < min and sptSet[v] == False: 
			min = dist[v] 
			min_index = v 

	return min_index 

# Funtion that implements Dijkstra's single source 
# shortest path algorithm for a graph represented 
# using adjacency matrix representation 
def dijkstra(src,graph,V): 

	dist = [l] * V 
	dist[src] = 0
	sptSet = [False] * V 

	for cout in range(V): 

		# Pick the minimum distance vertex from 
		# the set of vertices not yet processed. 
		# u is always equal to src in first iteration 
		u = minDistance(dist, sptSet,V) 

		# Put the minimum distance vertex in the 
		# shotest path tree 
		sptSet[u] = True

		# Update dist value of the adjacent vertices 
		# of the picked vertex only if the current 
		# distance is greater than new distance and 
		# the vertex in not in the shotest path tree 
		for v in range(V): 
			if graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]: 
					dist[v] = dist[u] + graph[u][v] 

	return dist












t=int(input())
while t>0:
    t=t-1
    a=input().split()
    n,m=int(a[0]),int(a[1])
    x=[]
    y=[]
    c=[]
    for i in range(m):
        a=input().split()
        #print(a)
        x.append(int(a[0]))
        y.append(int(a[1]))
        c.append(int(a[2]))
        
    q=int(input())
    h=[]
    k=[]
    for i in range(q):
        a=input().split()
        h.append(int(a[0]))
        k.append(int(a[1]))
    graph=[]
    graph = [[l for column in range(n)] 
			    for row in range(n)] 
    for i in range(m):
        graph[x[i]-1][y[i]-1]=c[i]
    dist=dijkstra(0,graph,n)
    #print(dist)
    for i in range(q):
        ans=k[i]-2*dist[h[i]-1]
        if ans<=0:
            print(0)
        else:
            print(ans)
    

        
    