from qiskit import(QuantumCircuit, execute, Aer)
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram, circuit_drawer

import sys
import numpy as np
import itertools
import math
import time
import datetime
import matplotlib.pyplot as plt 
import func
from func import *
   
        
# U_f | x > | b > = | x > | b + f(x) >
# preserve the state of the first n qubits
# compute all possible bit strings of size 2*n and create dictionary of each input paired with its index
# create another dictionary of each output value (f(x) mod2 b) paired with its index
# the i-th diagonal block corresponds to the i-th possible n-bit input x_i to f
# The indices mapped to the input values represent the rows of U_f. 
# Indices mapped to the output values represent the columns of U_f.
# If the input and output of indices a and b are equal, the U_f[a][b] = 1    

def get_U_f(f,n):

    U_f = np.zeros((2** (2*n), 2 **(2*n)))

    input_dict = {}
    output_dict = {}
    for idx, inputs in enumerate(list(itertools.product([0,1], repeat = 2*n))):
        
        input_dict.update({inputs: idx})

        func_input = inputs[0:n]
        b = inputs[n: 2*n]
        func_output = f(func_input)
       
        #f(x) mod b 
        inputs = list(inputs)
        for j in range(n):
            if func_output[j] != b[j]:
                inputs[n+j] = 1
            else:
                inputs[n+j] = 0
        
        inputs = tuple(inputs)


        #note, first n bits of inputs remain unchanged

        output_dict.update({inputs: idx})
    
    for y, ind1 in output_dict.items():
        for x, ind2 in input_dict.items():
            if x == y:
                U_f[ind2][ind1] = 1
                break
 
    return U_f

def simon_program(U_f, n):
    circuit = QuantumCircuit(2*n, n)

    # apply Hadamard to n qubits
    for i in range(n):
        circuit.h(i)

    # define the U_f gate based on the unitary matrix returned by get_U_f
    U_f_GATE = Operator(U_f)
    circuit.unitary(U_f_GATE, range(2*n-1,-1,-1), label='U_f')
    
    # apply Hadamard to all input qubits
    for k in range(n):
        circuit.h(k)

    #print(circuit_drawer(circuit, output='text'))
    return circuit


#returns nontrivial rows of rref of matrix of equations
def bitwise_gauss_elim(a):
    a = np.array(a)
    a %= 2
    a = a[a[:,0].argsort()][::-1]
    n = len(a[0])

    for i in range(n):
        ind = i
        while not a[i][ind]:
            ind += 1
            if ind >= n:
                return a[:i]

        for j in range(len(a)):
            if j == i:
                continue
            if a[j][ind]:
                a[j] += a[i]
                a[j] %= 2

        for k in range(n-1, -1, -1):
            a = a[::-1]
            a = a[a[:,k].argsort(kind='mergesort')][::-1]

    return a[:n]


#solve system of y's to determine s
#list_y contains y's, n-1 of which are linearly independent
def constraint_solver(list_y, n):
    # we need to solve the linear system given by y . s = 0, for n-1 equations
    s = [1] * n
    ind_y = bitwise_gauss_elim(list_y)
    for y in ind_y:
        # if y has more than one 1
        if np.sum(y)%2  == 0:
            continue
        
        for j in range(n):
            if y[j]:
                s[j] = 0
                break
    #s only contains 1s in the indices all the other y's contain 0s

    return s 

def print_results(test_name, result, exec_time, n):

    print()
    print()
    print('===================================')
    print()
    print('n value', n)
    print('Execution time:', exec_time, 'sec')
    print()
    print('===================================')
    print('===================================')
    print()

    for t in range(trials):
        vedict = None
        print('===================================')
        print()
        print('Trial', t + 1)
        
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
        print('\nLook in func.py for a function name to pass in as an argument, followed by the length of the bit string and the number of trials.\nRefer to README for additional info.\n')
        exit()

    func_in_name = sys.argv[1]
    try:
        func_in = getattr(func, func_in_name)
    except AttributeError:
        raise NotImplementedError("Class `{}` does not implement `{}`".format(func.__class__.__name__, func_in_name))
    
  
    n = int(sys.argv[2])
    trials = int(sys.argv[3]) # number of 4-cycle iterations of circuit

    simulator = Aer.get_backend('qasm_simulator')
   
    allzero_y = []
    exec_times = []
    for n_test in [n]:

        rank = 0
        list_y = []
        start_time = time.time()
        for j in range(4*trials):
            for iter in range(n_test):
                
                U_f = get_U_f(func_in, n_test)
                circuit = simon_program(U_f, n)
                circuit.measure(range(n), range(n))
                job = execute(circuit, simulator, shots=1)
                results = job.result()
                counts = results.get_counts(circuit)
                result_string = ''
                for x in counts:
                    if counts[x]:
                        result_string = x
                result = []
                for a in result_string:
                    result.append(int(a))
                result = result[::-1]

                list_y.append(result)

            
    #solve for s
    s_test = constraint_solver(list_y, n_test)
    if func_in(s_test) == func_in([0]*n):
        s = s_test
    else:
        s = [0]*n
    timepassed = time.time() - start_time
    exec_times.append(timepassed)

    print('s_test: ', s)
