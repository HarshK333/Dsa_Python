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

