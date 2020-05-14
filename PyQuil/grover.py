from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime

import sys
import numpy as np
import itertools
import math
import time
import datetime
import matplotlib.pyplot as plt 
import func
from func import *

# Z_f is just a 2^n by 2^n diagonal matrix
# the i-th entry in the diagonal corresponds to the 2^i-th possible n-bit input x and is equal to (-1)^f(x)
# hence, | x > gets mapped to (-1)^f(x) | x >
def get_Z_f(f, n):
	return np.diag([(-1) ** (f(x)) for x in list(itertools.product([0, 1], repeat=n))])

# Z_0 is just a 2^n by 2^n diagonal matrix
# the i-th entry in the diagonal corresponds to the 2^i-th possible n-bit input x and is equal to -1 if x == {0}^n and 1 otherwise
# hence, | x > gets mapped to -| x > if x == {0}^n and | x > otherwise
def get_Z_0(n):
	Z_0 = np.eye(2 ** n)
	Z_0[0, 0] = -1
	return Z_0

# generates the quantum circuit for Grover's algorithm given Z_f, Z_0, and n (the length of input bit strings)
def grover_program(Z_f, Z_0, n):
	p = Program()

	# apply Hadamard to all qubits
	for i in range(n):
		p += H(i)

	# define the Z_f gate based on the unitary matrix returned by get_Z_f
	Z_f_def = DefGate("Z_f", Z_f)
	Z_f_GATE = Z_f_def.get_constructor()
	p += Z_f_def

	# define the Z_0 gate based on the unitary matrix returned by get_Z_0
	Z_0_def = DefGate("Z_0", Z_0)
	Z_0_GATE = Z_0_def.get_constructor()
	p += Z_0_def

	# define the neg_I gate, which flips the sign of a qubit
	# achieves multiplication by -1 in the G operator (as defined in the lecture notes)
	# not necessary, since global phases are irrelevant, but just for consistency with lecture notes
	neg_I_def = DefGate("neg_I", -1 * np.eye(2))
	NEG_I_GATE = neg_I_def.get_constructor()
	p += neg_I_def

	# iterate floor of pi/4 * sqrt(2^n) times, as prescribed
	# on each iteration, apply the G operator
	for i in range(math.floor(math.pi / 4 * math.sqrt(2 ** n))):
		# apply Z_f gate
		p += Z_f_GATE(*range(n))

		# apply Hadamard to all qubits
		for j in range(n):
			p += H(j)

		# apply Z_0 gate
		p += Z_0_GATE(*range(n))

		for j in range(n):
			p += H(j)

		# take negative
		p += NEG_I_GATE(0)

	# print(p)
	return p

# pretty print results
def print_results(test_name, func, result, exec_time, trials, n):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
	print('Execution time:', exec_time, 'sec')
	print()
	print('===================================')
	print('===================================')
	print()

	for t in range(trials):
			print('===================================')
			print()
			print('Trial', t + 1)
			print('x =', ''.join([str(result[i][t]) for i in range(n)]))
			print('f(x) =', func([result[i][t] for i in range(n)]))
			print()
			print('===================================')
			print()
			print()

# Discuss your effort to test the two programs and present results from the testing.
# Discuss whether different cases of U_f lead to different execution times.
# What is your experience with scalability as n grows?  Present a diagram that maps n to execution time.

# test driver
if __name__ == '__main__':

	if len(sys.argv) <= 1:
		print('\nLook in func.py for a function name to pass in as an argument, followed by the length of the bit string and the number of trials.\nAlternatively, pass in the function name followed by \'--graph\' to create of graph of the scalability of the chosen function.\nRefer to README for additional info.\n')
		exit()
	graph = False
	if sys.argv[2] == '--graph':
		graph = True
	func_in_name = sys.argv[1]
	try:
	    func_in = getattr(func, func_in_name)
	except AttributeError:
	    raise NotImplementedError("Class `{}` does not implement `{}`".format(func.__class__.__name__, func_in_name))
	if not graph:
		n = int(sys.argv[2])
		trials = int(sys.argv[3])

	with local_forest_runtime():
		qc = get_qc('16q-qvm', noisy = False)
		qc.compiler.client.timeout = 10000

		all_funcs = [(func_in, func_in_name)] #, \
										# (all_ones, "All 1's"), \
										# (all_zeros, "All 0's"), \
										# (xnor, 'XNOR-reduce'), \
										# (zero, 'Constant 0')]

		if not graph:
			for fn, fn_name in all_funcs:
				Z_f = get_Z_f(fn, n)
				Z_0 = get_Z_0(n)
				start_time = time.time()
				p = grover_program(Z_f, Z_0, n)
				result = qc.run_and_measure(p, trials=trials)
				print_results(fn_name, fn, result, time.time() - start_time, trials, n)

		if graph:
			for fn, fn_name in all_funcs:
				exec_times = []
				for n_test in [int(arg) for arg in sys.argv[3:]]:
					Z_f = get_Z_f(fn, n_test)
					Z_0 = get_Z_0(n_test)
					start_time = time.time()
					p = grover_program(Z_f, Z_0, n_test)
					result = qc.run_and_measure(p, trials=1)
					exec_times.append(time.time() - start_time)

				plt.figure()
				plt.plot([int(arg) for arg in sys.argv[3:]], exec_times)
				plt.xlabel('Number of qubits')
				plt.ylabel('Execution time (in seconds)')
				plt.title('Scalability as number of qubits grows for Grover on %s' % fn_name)
				plt.savefig('grover_scalability_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % fn_name)

