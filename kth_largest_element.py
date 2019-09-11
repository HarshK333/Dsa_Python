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

