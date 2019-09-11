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