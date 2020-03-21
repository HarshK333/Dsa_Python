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