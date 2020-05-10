from pyquil import Program, get_qc
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime

import sys
import numpy as np
import itertools
import math

def f(x):
	return int(sum(x) == len(x))

def g(x):
	return int(sum(x) == 0)

def zero(x):
	return 0

def get_Z_f(f, n):
	return np.diag([(-1) ** (f(x)) for x in list(itertools.product([0, 1], repeat=n))])

def get_Z_0(n):
	Z_0 = np.eye(2 ** n)
	Z_0[0, 0] = -1
	return Z_0

def grover_program(Z_f, Z_0, n):
	p = Program()
	for i in range(n):
		p += H(i)

	Z_f_def = DefGate("Z_f", Z_f)
	Z_f_GATE = Z_f_def.get_constructor()
	p += Z_f_def

	Z_0_def = DefGate("Z_0", Z_0)
	Z_0_GATE = Z_0_def.get_constructor()
	p += Z_0_def

	neg_I_def = DefGate("neg_I", -1 * np.eye(2 ** n))
	NEG_I_GATE = neg_I_def.get_constructor()
	p += neg_I_def

	for i in range(math.floor(math.pi / 4 * math.sqrt(2 ** n))):
		p += Z_f_GATE(*tuple(range(n)))
		for j in range(n):
			p += H(j)
		p += Z_0_GATE(*tuple(range(n)))
		for j in range(n):
			p += H(j)
		p += NEG_I_GATE(*tuple(range(n)))

	print(p)
	return p

# Discuss your effort to test the two programs and present results from the testing.  Discuss whether different cases of U_f lead to different execution times.
# What is your experience with scalability as n grows?  Present a diagram that maps n to execution time.

if __name__ == '__main__':
	n = int(sys.argv[1])

	with local_forest_runtime():
		qc = get_qc('9q-square-qvm')

		p = grover_program(get_Z_f(f, n), get_Z_0(n), n)
		result = qc.run_and_measure(p, trials=10)
		for i in range(n):
			print(result[i])

		p = grover_program(get_Z_f(g, n), get_Z_0(n), n)
		result = qc.run_and_measure(p, trials=10)
		for i in range(n):
			print(result[i])

		p = grover_program(get_Z_f(zero, n), get_Z_0(n), n)
		result = qc.run_and_measure(p, trials=10)
		for i in range(n):
			print(result[i])

