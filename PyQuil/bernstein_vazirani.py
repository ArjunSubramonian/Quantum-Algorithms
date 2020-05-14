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
# i.e., it is X^f(x_i)
def get_U_f(f, n):
	U_f = np.zeros((2 ** (n + 1), 2 ** (n + 1)))
	for idx, inputs in enumerate(list(itertools.product([0, 1], repeat=n))):
		output = f(inputs)
		
		# the 2x2 box on the diagonal is I
		if output == 0:
			U_f[2 * idx, 2 * idx] = 1
			U_f[2 * idx + 1, 2 * idx + 1] = 1
		
		# the 2x2 box on th diagonal is Pauli X
		elif output == 1:
			U_f[2 * idx, 2 * idx + 1] = 1
			U_f[2 * idx + 1, 2 * idx] = 1

	return U_f

# generates the quantum circuit for Bernstein-Vazirani given U_f and n (the length of input bit strings)
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

	#print(p)
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


if __name__ == '__main__':

	if len(sys.argv) <= 1 or len(sys.argv) > 4:
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
		
		if not graph:
			b = func_in([0]*n)
			U_f = get_U_f(func_in, n)
			start_time = time.time()
			p = bv_program(U_f, n)
			result = qc.run_and_measure(p, trials=trials)
			print_results(func_in_name, result, time.time() - start_time, trials, n, b)
		
		if graph:
			exec_times = []
			for n_test in [1, 2, 3, 4]:
				U_f = get_U_f(func_in, n_test)
				start_time = time.time()
				p = bv_program(U_f, n_test)
				result = qc.run_and_measure(p, trials=1)
				exec_times.append(time.time() - start_time)
			plt.figure()
			plt.plot([1,2,3,4], exec_times)
			plt.xlabel('Number of Qubits')
			plt.ylabel('Execution time (sec)')
			plt.title('Scalability of Bernstein-Vazirani on %s' % func_in_name)
			plt.savefig('bernstein_vazirani_scalability_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name)	
