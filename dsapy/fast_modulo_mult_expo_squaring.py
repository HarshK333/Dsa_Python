def fast_exp(base,exp):
	res=1
	MOD=10**9+7
	while(exp>0):
		if(exp%2==1):
			res=(res*base)%MOD
		base=(base*base)%MOD
		exp=exp//2
	return res%MOD

#driver code 
base=int(input("base:"))
expo=int(input("expo:"))
print(fast_exp(base, expo))