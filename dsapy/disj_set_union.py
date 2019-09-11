
  
# Creates n sets with single item in each 
def makeSet(n,parent):
    for i in range(n):
        print(i) 
        parent[i]=i

# Returns representative of x 
def find(x,parent): 
    if parent[x] != x:
        parent[x] = find(parent[x],parent)  
    return parent[x] 

# Unites the set that includes x and the set 
# that includes y

def union(x, y,parent,rank):
    xRoot = find(x,parent)
    yRoot = find(y,parent)  
    if xRoot == yRoot: 
        return
    if rank[xRoot] < rank[yRoot]: 
        parent[xRoot] = yRoot 
    elif rank[yRoot] < rank[xRoot]:
        parent[yRoot] = xRoot   
    else:
        parent[yRoot] = xRoot 
        rank[xRoot] = rank[xRoot] + 1 

n=int(input("number of elements:"))
parent=[0]*n
rank=[0]*n
print(parent)
makeSet(n,parent)
q=int(input("number of queries:"))
for i in range(q):
    x=int(input("element:"))
    y=int(input("element:"))
    typ=int(input("type:"))
    if typ==0:
        union(x,y,parent,rank)
    else:
        xp=find(x,parent)
        yp=find(y,parent)
        if xp==yp:
            print("True")
        else:
            print("False")




  


     
 