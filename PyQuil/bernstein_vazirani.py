from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime

import sys
import numpy as np
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import func
from func import *


# U_f | x > | b > = | x > | b + f(x) >
# preserve the state of the first n qubits
# if f(x) == 0, we preserve the state of the helper qubit b
# otherwise, if f(x) == 1, we invert the state of the helper qubit
# U_f is just a 2^n by 2^n matrix where the "diagonal" consists of 2 by 2 blocks that are either the identity or Pauli X
# the i-th diagonal block corresponds to the i-th possible n-bit input x_i to f
## it is I if f(x_i) == 0
## it is X if f(x_i) == 1
def get_U_f(f, n):
	U_f = np.zeros((2 ** (n + 1), 2 ** (n + 1)))
	for idx, inputs in enumerate(list(itertools.product([0, 1], repeat=n))):
		output = f(inputs)
		if output == 0:
			U_f[2 * idx, 2 * idx] = 1
			U_f[2 * idx + 1, 2 * idx + 1] = 1
		elif output == 1:
			U_f[2 * idx, 2 * idx + 1] = 1
			U_f[2 * idx + 1, 2 * idx] = 1

	return U_f

# generates the quantum circuit for Deutsch-Jozsa given U_f and n (the length of input bit strings)
def bv_program(U_f, n):
	p = Program()

	# invert the helper qubit to make it 1
	p += X(n)
	# apply Hadamard to all input qubits and helper qubit
	for i in range(n + 1):
		p += H(i)

	# define the U_f gate based on the unitary matrix returned by get_U_f
	U_f_def = DefGate("U_f", U_f)
	U_f_GATE = U_f_def.get_constructor()
	p += U_f_def
	p += U_f_GATE(*range(n + 1))
	
	# apply Hadamard to all input qubits
	for i in range(n):
		p += H(i)

	# print(p)
	return p

# pretty print results
def print_results(test_name, result, exec_time, trials, n, b):

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
			verdict = None

			print('===================================')
			print()
			print('Trial', t + 1)
			print()
			print('a is ', *[result[i][t] for i in range(n)], sep = '')
			print('b is', str(b))
			print()
			print('===================================')
			print()
			print()

# Discuss your effort to test the two programs and present results from the testing.
# Discuss whether different cases of U_f lead to different execution times.
# What is your experience with scalability as n grows?  Present a diagram that maps n to execution time.

# File "/Users/arjuns/Documents/UCLA/Classes/Spring 2020/CS 239/venv/lib/python3.6/site-packages/rpcq/_client.py", line 205, in call
#     raise utils.RPCError(reply.error)
# rpcq._utils.RPCError: Unhandled error in host program:
# The assertion
# (NOT
#  #1=(TYPEP (FIRST (CL-QUIL::GOVERNED-QUEUE-CONTENTS CL-QUIL::SUBGOVERNOR))
#            'CL-QUIL:APPLICATION))
# failed with #1# = T.

if __name__ == '__main__':

	if len(sys.argv) <= 1:
		print('Look in func.py for a function name to pass in as an argument, followed by the length of the bit string and the number of trials.')
		exit()
	func_in_name = sys.argv[1]
	try:
	    func_in = getattr(func, func_in_name)
	except AttributeError:
	    raise NotImplementedError("Class `{}` does not implement `{}`".format(func.__class__.__name__, func_in_name))
	n = int(sys.argv[2])
	trials = int(sys.argv[3])

	with local_forest_runtime():
		qc = get_qc('9q-square-qvm')
		qc.compiler.client.timeout = 10000
		b = func_in([0]*n)
		U_f = get_U_f(func_in, n)
		start_time = time.time()
		p = bv_program(U_f, n)
		result = qc.run_and_measure(p, trials=trials)
		print_results(func_in_name, result, time.time() - start_time, trials, n, b)
		'''
		exec_times = []
		for n_test in [1, 2, 3, 4]:
			U_f = get_U_f(func_in, n_test)
			start_time = time.time()
			p = bv_program(U_f, n_test)
			result = qc.run_and_measure(p, trials=1)
			exec_times.append(time.time() - start_time)
			
		print(*exec_times, sep = ' s\t')
		'''