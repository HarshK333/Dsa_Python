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



