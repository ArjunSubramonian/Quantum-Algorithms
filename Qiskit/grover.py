from qiskit import(
    QuantumCircuit,
    execute,
    Aer)
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info.operators import Operator
from inspect import signature

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
def grover_program(Z_f, Z_0, n, draw_circuit):

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
		circuit.unitary(Z_f_GATE, range(n-1, -1, -1), label = 'Z_f')

		# apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the Z_0 gate based on the unitary matrix returned by get_Z_0
		Z_0_GATE = Operator(Z_0)
		circuit.unitary(Z_f_GATE, range(n-1, -1, -1), label = 'Z_0')

        #apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the neg_I gate, which flips the sign of a qubit
	    # achieves multiplication by -1 in the G operator (as defined in the lecture notes)
	    # not necessary, since global phases are irrelevant, but just for consistency with lecture notes
		NEG_I_GATE = Operator(-1 * np.eye(2**n))
		circuit.unitary(NEG_I_GATE, range(n), label = 'neg_I_def')

	if draw_circuit:
		print(circuit_drawer(circuit, output='text'))
	return circuit

# pretty print results
def print_results(test_name, result, transpile_time, trials, n):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
	print('Transpile time:', trasnpile_time, 'sec')
	print('Run time:', result.time_taken, 'sec')
	print()
	print('===================================')
	print('===================================')
	print()

	counts = result.get_counts(circuit)
	counts_sorted = sorted(counts.items(), key=lambda item: item[1], reverse=True)

	for idx, (key,value) in enumerate(counts_sorted):
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
	# plt.savefig('grover_hist_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % test_name)	
	plt.savefig('grover_hist.png')

if __name__ == '__main__':

	if len(sys.argv) <= 2:
		print('\nLook in func.py for a function name to pass in as an argument, followed by the length of the bit string and the number of trials.\nAlternatively, pass in the function name followed by \'--graph\' to create of graph of the scalability of the chosen function.\nRefer to README for additional info.\n')
		exit(1)
	graph = False
	draw_circuit = False
	if sys.argv[2] == '--graph':
		graph = True
	elif sys.argv[2] == '--draw':
	    draw_circuit = True

	func_in_name = sys.argv[1]
	try:
	    func_in = getattr(func, func_in_name)
	except AttributeError:
	    raise NotImplementedError("Class `{}` does not implement `{}`".format(func.__class__.__name__, func_in_name))
	sig = signature(func_in)
	if len(sig.parameters) != 1:
		print('\nSpecified function must only accept a single parameter: a bit string passed in as a Python list. Refer to README for additional info.\n')  
		exit(1)

	if not graph and not draw_circuit:
		n = int(sys.argv[2])
		trials = int(sys.argv[3])
		if len(sys.argv) > 4:
			try:
				optimization_level = int(sys.argv[4])
				if optimization_level < 0 or optimization_level > 3:
					print('\nOptimization level must be an integer between 0 and 3, inclusive. Higher levels generate more optimized circuits, at the expense of longer transpilation time.\n')
					exit(1)
			except:
				print('\nOptimization level must be an integer between 0 and 3, inclusive. Higher levels generate more optimized circuits, at the expense of longer transpilation time.\n')
				exit(1)
		else:
			optimization_level = 1

	simulator = Aer.get_backend('qasm_simulator')

	
	if not graph and not draw_circuit:
		Z_f = get_Z_f(func_in, n)
		Z_0 = get_Z_0(n)
		circuit = grover_program(Z_f, Z_0, n, False)
		circuit.measure(range(n), range(n-1, -1, -1))
		start = time.time()

        # gates available on IBMQX5
		circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
		end = time.time()
		job = execute(circuit, simulator, shots=trials)
		print_results(func_in_name, job.result(), end - start, trials, n)

	if graph:
		transpile_times = [[], [], [], []]
		run_times = [[], [], [], []]
		qubits = []
		
        # if the no. of test qubits are specified
		if len(sys.argv) > 3:
			qubits = sorted(list(map(int, sys.argv[3:])))
		
	    # default is test on n = 1,2,3,4
		else:
		    qubits = [1,2,3,4]

		for optimization_level in range(4):
			for n_test in qubits:
				Z_f = get_Z_f(func_in, n_test)
				Z_0 = get_Z_0(n_test)
				circuit = grover_program(Z_f, Z_0, n_test, False)
				circuit.measure(range(n_test), range(n_test-1, -1, -1))

				start = time.time()
				# gates available on IBMQX5
				circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
				end = time.time()
				job = execute(circuit, simulator, optimization_level=0, shots=1)

				transpile_times[optimization_level].append(end - start)
				run_times[optimization_level].append(job.result().time_taken)
		

		for optimization_level in range(4):
			plt.figure()
			plt.plot(qubits, transpile_times[optimization_level])
			plt.xlabel('Number of Qubits')
			plt.ylabel('Transpile time (sec)')
			plt.title('Transpile time scalability of Grover on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('grover_transpile_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)

			plt.figure()
			plt.plot(qubits, run_times[optimization_level])
			plt.xlabel('Number of Qubits')
			plt.ylabel('Run time (sec)')
			plt.title('Run time scalability of Grover on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('grover_run_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)	

		fig, ax1 = plt.subplots()
		ax1.set_xlabel('Qiskit optimization level')
		ax1.set_ylabel('Transpile time (sec)')
		ln1 = ax1.plot(range(4), [transpile_times[i][-1] for i in range(4)], 'r', label='transpile')

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		ax2.set_ylabel('Run time (sec)') # we already handled the x-label with ax1
		ln2 = ax2.plot(range(4), [run_times[i][-1] for i in range(4)], 'b', label='run')

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.legend(ln1 + ln2, ['transpile', 'run'], loc=0)
		plt.subplots_adjust(top=0.88)
		plt.suptitle('Comparison of transpile and run times for Grover on %s\n(%d qubits)' % (func_in_name, qubits[-1]))
		plt.savefig('grover_run_transpile_comp_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name, fontsize=8)	

	if draw_circuit:
		grover_program(get_Z_f(func_in,  int(sys.argv[3])), get_Z_0(int(sys.argv[3])), int(sys.argv[3]), draw_circuit=True)