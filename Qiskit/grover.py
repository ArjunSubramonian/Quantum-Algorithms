from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info.operators import Operator

import math
import sys
import numpy as np
import itertools
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

	#q = QuantumRegister(n, 'q')
	#c = QuantumRegister(n, 'c')
	circuit = QuantumCircuit(n,n)

	# apply Hadamard to all qubits
	for i in range(n):
		circuit.h(i)

	# iterate floor of pi/4 * sqrt(2^n) times, as prescribed
	# on each iteration, apply the G operator
	for i in range(math.floor(math.pi / 4 * math.sqrt(2 ** n))):

		# define the Z_f gate based on the unitary matrix returned by get_Z_f
		Z_f_GATE = Operator(Z_f)
		circuit.unitary(Z_f_GATE, range(n), label = 'Z_f')

		# apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the neg_I gate, which flips the sign of a qubit
		Z_0_GATE = Operator(Z_0)
		circuit.unitary(Z_f_GATE, range(n), label = 'Z_0')

        #apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the neg_I gate, which flips the sign of a qubit
	    # achieves multiplication by -1 in the G operator (as defined in the lecture notes)
	    # not necessary, since global phases are irrelevant, but just for consistency with lecture notes
		NEG_I_GATE = Operator(-1 * np.eye(2**n))
		circuit.unitary(NEG_I_GATE, range(n), label = 'neg_I_def')

	print(circuit_drawer(circuit, output='text'))
	return circuit

# pretty print results
def print_results(test_name, result, trials, n):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
	print('Execution time:', result.time_taken, 'sec')
	print()
	print('===================================')
	print('===================================')
	print()

	counts = result.get_counts(circuit)

	for idx, key in enumerate(counts):
			print('===================================')
			print()
			print('Result', idx + 1)
			print('Frequency:', counts[key])
			print()
			print('a is', key)
			print()
			print('===================================')
			print()
			print()

	plot_histogram(counts, title='Test: ' + test_name)
	# plt.savefig('bernstein_vazirani_hist_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % test_name)	
	plt.savefig('grover_hist.png')

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

	simulator = Aer.get_backend('qasm_simulator')

	all_funcs = [(func_in, func_in_name)] #, \
										# (all_ones, "All 1's"), \
										# (all_zeros, "All 0's"), \
										# (xnor, 'XNOR-reduce'), \
										# (zero, 'Constant 0')]
	if not graph:
		for fn, fn_name in all_funcs:
			Z_f = get_Z_f(fn, n)
			Z_0 = get_Z_0(n)
			circuit = grover_program(Z_f, Z_0, n)
			circuit.measure(range(n), range(n)) # cbit?
			job = execute(circuit, simulator, shots=trials)
			result = job.result()
			print_results(fn_name, result, trials, n)

	if graph:
		for fn, fn_name in all_funcs:
			exec_times = []
			qubits = []

            # if the no. of test qubits are specified
			if len(sys.argv) > 3:
			    qubits = sorted(list(map(int, sys.argv[3:])))
		
		    # default is test on n = 1,2,3,4
			else:
			    qubits = [1,2,3,4,5,6]

			for n_test in qubits:
				Z_f = get_Z_f(fn, n_test)
				Z_0 = get_Z_0(n_test)
				circuit = grover_program(Z_f, Z_0, n_test)
				circuit.measure(range(n_test), range(n_test))
				job = execute(circuit, simulator, shots=1)
				result = job.result()
				exec_times.append(job.result().time_taken)

			plt.figure()
			plt.plot(qubits, exec_times)
			plt.xlabel('Number of qubits')
			plt.ylabel('Execution time (sec)')
			plt.title('Scalability as number of qubits grows for Grover on %s' % fn_name)
			plt.savefig('grover_scalability_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % fn_name)
