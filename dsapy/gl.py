a=5
def b():
	#global a
	global a
	a=2
	print(a)
	global b
	b=5
b()
print(a)
print(b)

