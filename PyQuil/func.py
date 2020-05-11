# constant 0 function
# Doesn't return 1 for any inputs (same as Constant 0)
def zero(x):
	return 0

# constant 1 function
def one(x):
	return 1

# XNOR-reduce
# balanced
def xnor(x):
	return sum(x) % 2

# XOR-reduce
# balanced
def xor(x):
	return (sum(x) + 1) % 2 

	# Returns 1 only in the case that x is an n-bit string of all 1's
def f(x):
	return int(sum(x) == len(x))

# Returns 1 only in the case that x is an n-bit string of all 0's
def g(x):
	return int(sum(x) == 0)

# Returns 1 when x has an odd number of 1's (same as XNOR-reduce)
def h(x):
	return sum(x) % 2