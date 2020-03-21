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
    







            




    
