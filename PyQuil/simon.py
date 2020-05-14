from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quil import DefGate

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
# U_f is just a 2^(2*n) by 2^(2*n) matrix where the "diagonal" consists of 2 by 2 blocks that are either the identity or Pauli X
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
 
    #print (U_f)
    return U_f

# generates the quantum circuit for Deutsch-Jozsa given U_f and n (the length of input bit strings)
def simon_program(U_f, n):
    p = Program()

	# apply Hadamard to n qubits
    for i in range(n):
        p += H(i)

	# define the U_f gate based on the unitary matrix returned by get_U_f
    U_f_def = DefGate("U_f", U_f)
    U_f_GATE = U_f_def.get_constructor()
    p += U_f_def
    p += U_f_GATE(*range(2*n))

	# apply Hadamard to all input qubits
    for k in range(n):
        p += H(k)

	# print(p)
    return p


#solve system of y's to determine s
#list_y contains y's, n-1 of which are linearly independent
def constraint_solver(list_y, n):
    # we need to solve the linear system given by y . s = 0, for n-1 equations
    s = [1] * n
    
    for i in list_y:
        for j in range(n):
            if i[j] == 1 and s[j] == 1:
                s[j] = 0
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
        trials = int(sys.argv[3]) # number of 4-cycle iterations of circuit

   
    with local_forest_runtime():
        qc = get_qc('9q-square-qvm')
        qc.compiler.client.timeout = 10000
        allzero_y = []
        if (not graph):
            exec_times = []
            for n_test in [n]: 

                rank = 0
                list_indep_y = []
                start_time = time.time()
                for j in range(4*trials):
                    for iter in range(n_test-1):
                
                        U_f = get_U_f(func_in, n_test)
                        p = simon_program(U_f, n_test)
                        result = qc.run_and_measure(p, 1)
                    
                        print(result)

                        #appending result to list of y's later to be used for solving for s
                        #only append if result is not all 0s
                        allzero = True
                        for h in range(n_test):
                            if result[h] == 1:
                                allzero = False
                                break

                        if allzero == False:
                            list_indep_y.append(result)
                        else:
                            allzero_y = result
                    
                    #create matrix of y's
                    y_matrix = np.array(list_indep_y)
                    rank = np.linalg.matrix_rank(y_matrix)

                    #stop iterating through loop setup once there are n-1 linearly independent y's
                    if rank == n_test-1:
                        break
            
            #solve for s
            if rank == 0:
                list_indep_y.append(allzero_y)
            s_test = constraint_solver(list_indep_y, n_test)
            timepassed = time.time() - start_time
            exec_times.append(timepassed)

            print_results(func_in_name, s_test, timepassed, n_test)

            print('s_test: ', s_test)

        if graph:
            exec_times = []
            trials = 5
            for n_test in sorted([int(arg) for arg in sys.argv[3:]]): 
                rank = 0
                list_indep_y = []
                start_time = time.time()
                for j in range(4*trials):
                    for iter in range(n_test-1):
                
                            U_f = get_U_f(func_in, n_test)
                            p = simon_program(U_f, n_test)
                            result = qc.run_and_measure(p, 1)

                            #appending result to list of y's later to be used for solving for s
                            #only append if result is not all 0s
                            allzero = True
                            for h in range(n_test):
                                if result[h] == 1:
                                    allzero = False
                                    break

                            if allzero == False:
                                list_indep_y.append(result)
                            else:
                                allzero_y = result
                    
                    #create matrix of y's
                    y_matrix = np.array(list_indep_y)
                    rank = np.linalg.matrix_rank(y_matrix)

                    #stop iterating through loop setup once there are n-1 linearly independent y's
                    if rank == n_test-1:
                        break
            
                #solve for s
                if rank == 0:
                    list_indep_y.append(allzero_y)
                s_test = constraint_solver(list_indep_y, n_test)
                timepassed = time.time() - start_time
                exec_times.append(timepassed)

                print('s_test: ', s_test)
            plt.figure()
            plt.plot(sorted([int(arg) for arg in sys.argv[3:]]), exec_times)
            plt.xlabel('Number of qubits')
            plt.ylabel('Execution time (in seconds)')
            plt.title('Scalability as number of qubits grows for Simon on %s' % func_in_name)
            plt.savefig('simon_scalability_%s_{:%Y-%m-%d_%H-%M-%S}.png'.format(datetime.datetime.now()) % func_in_name)
