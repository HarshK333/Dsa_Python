# Python3 program for Naive Pattern  
# number of comparisons in the worst case is O(m*(n-m+1))
# KMP matching algorithm improves the worst case to O(n)
def search(pat, txt): 
    M = len(pat) 
    N = len(txt) 
  
    # A loop to slide pat[] one by one */ 
    for i in range(N - M + 1): 
        j = 0
          
        # For current index i, check  
        # for pattern match */ 
        while(j < M): 
            if (txt[i + j] != pat[j]): 
                break
            j += 1
  
        if (j == M):  
            print("Pattern found at index ", i) 
  
# Driver Code 
if __name__ == '__main__': 
    txt = "AABAACAADAABAAABAA"
    pat = "AABA"
    search(pat, txt) 

#Best case if first letter of pattern is not present
#worst case if all characters of pattern are everywhere in string or only last character differs
  