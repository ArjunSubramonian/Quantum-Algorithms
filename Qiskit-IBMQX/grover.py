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
import math
import sys
import numpy as np
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import func
from func import *

MY_API_KEY = ''
IBMQ.save_account(MY_API_KEY)
provider = IBMQ.load_account()
small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                   and not x.configuration().simulator)
backend = least_busy(small_devices)
#backend = provider.get_backend('ibmq_london')
simulator = Aer.get_backend('qasm_simulator')

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

	Z_f_GATE = Operator(Z_f)

	Z_0_GATE = Operator(Z_0)

	NEG_I_GATE = Operator(-1 * np.eye(2**n))

	# iterate floor of pi/4 * sqrt(2^n) times, as prescribed
	# on each iteration, apply the G operator
	for i in range(math.floor(math.pi / 4 * math.sqrt(2 ** n))):
		# define the Z_f gate based on the unitary matrix returned by get_Z_f
		#Z_f_GATE = Operator(Z_f)
		circuit.unitary(Z_f_GATE, range(n-1, -1, -1), label = 'Z_f')

		# apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the Z_0 gate based on the unitary matrix returned by get_Z_0
		#Z_0_GATE = Operator(Z_0)
		circuit.unitary(Z_f_GATE, range(n-1, -1, -1), label = 'Z_0')

        #apply Hadamard to all qubits
		for j in range(n):
			circuit.h(j)

		# define the neg_I gate, which flips the sign of a qubit
	    # achieves multiplication by -1 in the G operator (as defined in the lecture notes)
	    # not necessary, since global phases are irrelevant, but just for consistency with lecture notes
		#NEG_I_GATE = Operator(-1 * np.eye(2**n))
		circuit.unitary(NEG_I_GATE, range(n), label = 'neg_I_def')

		return circuit

# pretty print results
def print_results(test_name, circuit_size, results, meas_filter, transpile_time, trials, n):

	print()
	print()
	print('===================================')
	print()
	print('Test:', test_name)
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
			print('===================================')
			print()
			print('Result', idx + 1)
			print('Frequency:', counts[key])
			print('Mitigated frequency:', mitigated_counts[key])
			print()
			print('a is', key)
			print()
			print('===================================')
			print()
			print()

	plot_histogram([counts, mitigated_counts], title=test_name, legend=['raw', 'mitigated'])
	plt.axhline(1/(2 ** n), color='k', linestyle='dashed', linewidth=1)
	plt.savefig('grover_hist_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % test_name, bbox_inches = "tight")	

if __name__ == '__main__':

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
		Z_f = get_Z_f(func_in, n)
		Z_0 = get_Z_0(n)
		circuit = grover_program(Z_f, Z_0, n)
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
		print_results(func_in_name, circuit.size(), delayed_results, meas_fitter.filter, end - start, trials, n)


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
				Z_f = get_Z_f(func_in, n_test)
				Z_0 = get_Z_0(n_test)
				circuit = grover_program(Z_f, Z_0, n_test)
				circuit.measure(range(n_test), range(n_test - 1, -1, -1))

                #simulator transpile 

				start = time.time()
                # gates available on IBMQX5
				circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=optimization_level)
				end = time.time()
				job = execute(circuit, simulator, optimization_level=0, shots=1)
				
				sim_transpile_times[optimization_level].append(end - start)
				sim_run_times[optimization_level].append(job.result().time_taken)
				sim_gates[optimization_level].append(circuit.size())
				
                #quantum computer transpile
				start = time.time()
				circuit = transpile(circuit, backend, optimization_level=optimization_level)
				end = time.time()
				job = execute(circuit, backend, optimization_level=0, shots=1)
				delayed_result = backend.retrieve_job(job.job_id()).result()

				qc_transpile_times[optimization_level].append(end - start)
				qc_run_times[optimization_level].append(job.result().time_taken)
				qc_gates[optimization_level].append(circuit.size())

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

			plt.suptitle('Transpile time scalability of Grover on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('grover_transpile_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)

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

			plt.suptitle('Run time scalability of Grover on %s\n(optimization level = %d)' % (func_in_name, optimization_level))
			plt.savefig('grover_run_scalability_%s_%dopt_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % (func_in_name, optimization_level), fontsize=8)	

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
		plt.suptitle('Comparison of transpile and run times for Grover on %s\n(%d qubits)' % (func_in_name, qubits[-1]))
		plt.savefig('grover_run_transpile_comp_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name, fontsize=8)	


	
