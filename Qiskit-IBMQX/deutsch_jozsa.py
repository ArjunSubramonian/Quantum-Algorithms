from qiskit import(
  QuantumCircuit,
  execute,
  Aer,
  IBMQ,
  transpile,
  assemble)
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info.operators import Operator
# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

from inspect import signature
import sys
import numpy as np
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import func
from func import *

# MY_API_KEY = '' <-- ENTER KEY
# IBMQ.save_account(MY_API_KEY)
provider = IBMQ.load_account()
small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator)
backend = least_busy(small_devices)
#backend = provider.get_backend('ibmq_london')

simulator = Aer.get_backend('qasm_simulator')

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

# generates the quantum circuit for Deutsch-Jozsa given U_f and n (the length of input bit strings)
def dj_program(U_f, n):
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

	return circuit

# pretty print results
def print_results(test_name, circuit_size, results, meas_filter, transpile_time, trials, n, b):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
	print('Transpile time:', transpile_time, 'sec')
	print('Number of gates in transpiled circuit:', circuit_size)
	print('Run time:', sum([result.time_taken for result in results]) / trials, 'sec')
	print()
	print('===================================')
	print('===================================')
	print()

	# Compute counts and error-mitigated counts
	counts = {}
	mitigated_counts = {}
	for result in results:
		c = result.get_counts()
		mitigated_results = meas_filter.apply(result)
		mc = mitigated_results.get_counts(0)

		counts = {key: counts.get(key, 0) + c.get(key, 0) for key in set(counts) | set(c)}
		mitigated_counts = {key: mitigated_counts.get(key, 0) + mc.get(key, 0) for key in set(mitigated_counts) | set(mc)}
		
	counts_sorted = sorted(counts.items(), key=lambda item: item[1], reverse=True)
	for idx, (key, value) in enumerate(counts_sorted):
			verdict = 'Constant!'
			print('===================================')
			print()
			print('Result', idx + 1)
			print('Frequency:', counts[key])
			print('Mitigated frequency:', mitigated_counts[key])


            # Constant function if measure all 0's, balanced otherwise
			for i in range(n):
			    if key[i] != '0': 
			        verdict = 'Balanced!'
			        break

			print('Measurement:', key)
			print('Function is:', verdict)
			print()
			print()
			print('===================================')
			print()
			print()

	plot_histogram([counts, mitigated_counts], title=test_name, legend=['raw', 'mitigated'])
	plt.axhline(1/(2 ** n), color='k', linestyle='dashed', linewidth=1)
	plt.savefig('deutsch_jozsa_hist_%s_%d_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (test_name, trials),  bbox_inches = "tight")	

if __name__ == '__main__':

	# Process options and arguments
	if len(sys.argv) <= 2:
		print('\nLook in func.py for a function name to pass in as an argument, followed by the length of the bit string and the number of trials.\nAlternatively, pass in the function name followed by \'--graph\' to create of graph of the scalability of the chosen function.\nRefer to README for additional info.\n')
		exit(1)
	graph = False
	if sys.argv[2] == '--graph':
		graph = True

	func_in_name = sys.argv[1]
	try:
		func_in = getattr(func, func_in_name)
	except AttributeError:
		raise NotImplementedError("Class `{}` does not implement `{}`".format(func.__class__.__name__, func_in_name))
	sig = signature(func_in)
	if len(sig.parameters) != 1:
		print('\nSpecified function must only accept a single parameter: a bit string passed in as a Python list. Refer to README for additional info.\n')  
		exit(1)

	if not graph:
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
	
	if not graph:
		b = func_in([0]*n)
		U_f = get_U_f(func_in, n)

		circuit = dj_program(U_f, n)
		circuit.measure(range(n), range(n - 1, -1, -1))

		# Calibration matrix
		meas_calibs, state_labels = complete_meas_cal(qubit_list=range(n), circlabel='mcal')
		job = execute(meas_calibs, backend=backend, shots=8192)
		cal_results = job.result()
		meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')

		# Time transpilation separately
		start = time.time()
		circuit = transpile(circuit, backend, optimization_level=optimization_level)
		end = time.time()

		# Account for more than 8192 trials
		if trials > 8192:
			trials_list = [8192] * (trials // 8192) + [trials % 8192]
		else:
			trials_list = [trials]
		jobs = []
		for t in trials_list:
			jobs.append(execute(circuit, backend, optimization_level=0, shots=t))
		delayed_results = []
		for j in jobs:
			delayed_results.append(backend.retrieve_job(j.job_id()).result())
		print_results(func_in_name, circuit.size(), delayed_results, meas_fitter.filter, end - start, trials, n, b)
	
	if graph:
		sim_transpile_times = [[], [], [], []]
		sim_run_times = [[], [], [], []]
		sim_gates = [[], [], [], []]
		qc_transpile_times = [[], [], [], []]
		qc_run_times = [[], [], [], []]
		qc_gates = [[], [], [], []]
		qubits = []
		
		# if the no. of test qubits are specified
		if len(sys.argv) > 3:
			qubits = sorted(list(map(int, sys.argv[3:])))
		
		# default is test on n = 1,2,3
		else:
			qubits = [1,2,3]
		
		for optimization_level in range(4):
			for n_test in qubits:
					U_f = get_U_f(func_in, n_test)

					circuit = dj_program(U_f, n_test)
					circuit.measure(range(n_test), range(n_test - 1, -1, -1))

					start = time.time()
					# gates available on IBMQX5
					circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
					end = time.time()
					job = execute(circuit, simulator, optimization_level=0, shots=1)

					sim_transpile_times[optimization_level].append(end - start)
					sim_run_times[optimization_level].append(job.result().time_taken)
					sim_gates[optimization_level].append(circuit.size())

					start = time.time()
					circuit = transpile(circuit, backend, optimization_level=optimization_level)
					end = time.time()
					job = execute(circuit, backend, optimization_level=0, shots=1)
					delayed_result = backend.retrieve_job(job.job_id()).result()

					qc_transpile_times[optimization_level].append(end - start)
					qc_run_times[optimization_level].append(job.result().time_taken)
					qc_gates[optimization_level].append(circuit.size())

        #for graphing, adjust array of qubit values to include the helper qubit
		for i in range(len(qubits)):
			qubits[i] += 1

		for optimization_level in range(4):
			fig, ax1 = plt.subplots()
			ln11 = ax1.plot(qubits, sim_transpile_times[optimization_level], 'r', label="QASM simulator")
			ln12 = ax1.plot(qubits, qc_transpile_times[optimization_level], 'k', label=backend.name())
			ax1.set_xlabel('Number of Qubits')
			ax1.set_ylabel('Transpile time (sec)')

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
			ax2.set_ylabel('Number of gates') # we already handled the x-label with ax1
			ln21 = ax2.plot(qubits, sim_gates[optimization_level], 'b', label='QASM simulator')
			ln22 = ax2.plot(qubits, qc_gates[optimization_level], 'g', label=backend.name())

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			plt.legend(ln11 + ln12 + ln21 + ln22, ['QASM simulator transpile', backend.name() + ' transpile', 'QASM simulator #g', backend.name() + ' #g'], loc=0)
			plt.subplots_adjust(top=0.88)

			plt.suptitle('Transpile time scalability of Deutsch-Jozsa on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('deutsch_jozsa_transpile_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)

			# ==========

			fig, ax1 = plt.subplots()
			ln11 = ax1.plot(qubits, sim_run_times[optimization_level], 'r', label="QASM simulator")
			ln12 = ax1.plot(qubits, qc_run_times[optimization_level], 'k', label=backend.name())
			ax1.set_xlabel('Number of Qubits')
			ax1.set_ylabel('Run time (sec)')

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
			ax2.set_ylabel('Number of gates') # we already handled the x-label with ax1
			ln21 = ax2.plot(qubits, sim_gates[optimization_level], 'b', label='QASM simulator')
			ln22 = ax2.plot(qubits, qc_gates[optimization_level], 'g', label=backend.name())

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			plt.legend(ln11 + ln12 + ln21 + ln22, ['QASM simulator run', backend.name() + ' run', 'QASM simulator #g', backend.name() + ' #g'], loc=0)
			plt.subplots_adjust(top=0.88)

			plt.suptitle('Run time scalability of Deutsch-Jozsa on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('deutsch_jozsa_run_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)	

		fig, ax1 = plt.subplots()
		ax1.set_xlabel('Qiskit optimization level')
		ax1.set_ylabel('Transpile time (sec)')
		ln11 = ax1.plot(range(4), [sim_transpile_times[i][-1] for i in range(4)], 'r', label='QASM simulator transpile')
		ln12 = ax1.plot(range(4), [qc_transpile_times[i][-1] for i in range(4)], 'k', label=backend.name() + ' transpile')

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		ax2.set_ylabel('Run time (sec)') # we already handled the x-label with ax1
		ln21 = ax2.plot(range(4), [sim_run_times[i][-1] for i in range(4)], 'b', label='QASM simulator run')
		ln22 = ax2.plot(range(4), [qc_run_times[i][-1] for i in range(4)], 'g', label=backend.name() + ' run')

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.legend(ln11 + ln12 + ln21 + ln22, ['QASM simulator transpile', backend.name() + ' transpile', 'QASM simulator run', backend.name() + ' run'], loc=0)
		plt.subplots_adjust(top=0.88)
		plt.suptitle('Comparison of transpile and run times for Deutsch_Jozsa on %s\n(%d qubits)' % (func_in_name, qubits[-1]))
		plt.savefig('deutsch_jozsa_run_transpile_comp_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name, fontsize=8)	
