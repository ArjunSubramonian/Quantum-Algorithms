# constant 0 function
# Doesn't return 1 for any inputs (same as Constant 0)
def zero(x):
	return 0

# constant 1 function
def one(x):
	return 1

# XNOR-reduce
# balanced
# Returns 1 when x has an odd number of 1's (same as XNOR-reduce)
def xnor(x):
	return sum(x) % 2

# XOR-reduce
# balanced
def xor(x):
	return (sum(x) + 1) % 2 

# Returns 1 only in the case that x is an n-bit string of all 1's
def all_ones(x):
	return int(sum(x) == len(x))

# Returns 1 only in the case that x is an n-bit string of all 0's
def all_zeros(x):
	return int(sum(x) == 0)


# The following functions were designed to be used in the Bernstein-Vazirani algorithm
# Additionally, the zero() and one() function from above also work

# Returns only first bit
def first_bit(x):
        return x[0]

# Returns only last bit
def last_bit(x):
	return x[-1]

#Returns sum of bit string plus 1 (mod 2)
def add_one(x):
        return (sum(x)+1)%2

#Returns sum of bit string (mod 2)
def add(x):
	return sum(x) % 2

#Returns sum of bottom half of bit string (mod 2)
def bottom_half(x):
	return sum(x[(len(x)//2):]) % 2

#Returns sum of top half of bit string (mod 2)
def top_half(x):
	return sum(x[:(len(x)//2)]) % 2
