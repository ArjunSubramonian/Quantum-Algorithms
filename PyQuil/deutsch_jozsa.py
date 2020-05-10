from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime

import sys
import numpy as np
import itertools
import math

def zero(x):
	return 0

def xor(x):
	return sum(x) % 2

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

def dj_program(U_f, n):
	p = Program()

	p += X(n)
	for i in range(n + 1):
		p += H(i)

	U_f_def = DefGate("U_f", U_f)
	U_f_GATE = U_f_def.get_constructor()
	p += U_f_def
	p += U_f_GATE(*tuple(range(n + 1)))
	
	for i in range(n):
		p += H(i)

	print(p)
	return p

def print_results(result, trials, n):
	for t in range(trials):
			verdict = None

			print('===================================')
			print()
			print('Trial', t + 1)
			for i in range(n):
				if result[i][t] != 0:
					verdict = 'Balanced!'
					break

			if verdict is None:
				print('Constant!')
			else:
				print(verdict)
			print()
			print('===================================')
			print()
			print()

# Discuss your effort to test the two programs and present results from the testing.  Discuss whether different cases of U_f lead to different execution times.
# What is your experience with scalability as n grows?  Present a diagram that maps n to execution time.

if __name__ == '__main__':
	n = int(sys.argv[1])

	with local_forest_runtime():
		qc = get_qc('9q-square-qvm')

		p = dj_program(get_U_f(zero, n), n)
		result = qc.run_and_measure(p, trials=10)
		print_results(result, 10, n)

		p = dj_program(get_U_f(xor, n), n)
		result = qc.run_and_measure(p, trials=10)
		print_results(result, 10, n)




		

