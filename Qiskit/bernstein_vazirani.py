from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram, circuit_drawer
from qiskit.quantum_info.operators import Operator
from inspect import signature

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
def bv_program(U_f, n, draw_circuit=False):
	circuit = QuantumCircuit(n + 1, n)

	# invert the helper qubit to make it 1
	circuit.x(n)
	
	# apply Hadamard to all input qubits and helper qubit
	for i in range(n + 1):
		circuit.h(i)

	# define the U_f gate based on the unitary matrix returned by get_U_f
	U_f_gate = Operator(U_f)
	circuit.unitary(U_f_gate, range(n, -1, -1), label='U_f')
	
	# apply Hadamard to all input qubits
	for i in range(n):
		circuit.h(i)

	if draw_circuit:
		print(circuit_drawer(circuit, output='text'))
	return circuit

# pretty print results
def print_results(test_name, result, transpile_time, trials, n, b):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
	print('Transpile time:', transpile_time, 'sec')
	print('Run time:', result.time_taken, 'sec')
	print()
	print('===================================')
	print('===================================')
	print()

	counts = result.get_counts(circuit)
	counts_sorted = sorted(counts.items(), key=lambda item: item[1], reverse=True)
	for idx, (key, value) in enumerate(counts_sorted):
			print('===================================')
			print()
			print('Result', idx + 1)
			print('Frequency:', counts[key])
			print()
			print('a is', key)
			print('b is', b)
			print()
			print('===================================')
			print()
			print()

	plot_histogram(counts, title='Test: ' + test_name)
	plt.savefig('bernstein_vazirani_hist_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % test_name)	

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
		b = func_in([0]*n)
		U_f = get_U_f(func_in, n)

		circuit = bv_program(U_f, n)
		circuit.measure(range(n), range(n - 1, -1, -1))

		start = time.time()
		# gates available on IBMQX5
		circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
		end = time.time()
		job = execute(circuit, simulator, optimization_level=0, shots=trials)
		print_results(func_in_name, job.result(), end - start, trials, n, b)
	
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
					U_f = get_U_f(func_in, n_test)

					circuit = bv_program(U_f, n_test)
					circuit.measure(range(n_test), range(n_test - 1, -1, -1))

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
			plt.title('Transpile time scalability of Bernstein-Vazirani on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('bernstein_vazirani_transpile_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)

			plt.figure()
			plt.plot(qubits, run_times[optimization_level])
			plt.xlabel('Number of Qubits')
			plt.ylabel('Run time (sec)')
			plt.title('Run time scalability of Bernstein-Vazirani on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('bernstein_vazirani_run_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)	

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
		plt.suptitle('Comparison of transpile and run times for Bernstein-Vazirani on %s\n(%d qubits)' % (func_in_name, qubits[-1]))
		plt.savefig('bernstein_vazirani_run_transpile_comp_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name, fontsize=8)	

	if draw_circuit:
		bv_program(get_U_f(func_in, int(sys.argv[3])), int(sys.argv[3]), draw_circuit=True)
