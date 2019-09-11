
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