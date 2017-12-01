import numpy as np
import math 
from scipy import optimize

m = 20
n = 4
h = 1
eps = 0
tau = -10.0
param = {'h': h, 'eps': eps, 'm': m, 'n': n, 'tau': tau}

input_array = np.ones((m, n))

#used to test the optimizer
def norm(x):
	x = np.asarray(x)
	return np.linalg.norm(x)	

#assume model_values is flattened to x
def total_variation2d(x, eps, h, m, n):
#def total_variation2d(model_values, param):
	model_values = x.astype(np.float32).reshape(m, n)
	tv = 0
	#print(model_values.shape)
	#m, n = model_values.shape
	#assume model_values is given as array of size mxn
	#print(m, n)
	for i in range(m):
		for j in range(n):
			#print(model_values[i, j])
			update = 0
			if i < m-1:
				update += (model_values[i+1, j] - model_values[i, j])**2
			if j < n-1:
				update += (model_values[i, j+1] - model_values[i, j])**2
			
			tv +=math.sqrt(update)
	tv*=1/h  
	return tv

def total_variation2d_inequality(model_values, param):
	#we want tv(m) <= tau to be written as g(m) >= 0
	h = param['h']
	m = param['m']
	n = param['n']
	eps = param['eps']
	tau = param['tau']
	return -(total_variation2d(model_values, eps, h, m, n) - tau)


#tv = total_variation2d(input_array, eps, h)
x = input_array.flatten().astype(np.float64)
tv = total_variation2d(x, eps, h, m, n)
tvminustau = total_variation2d_inequality(x, param)

print(tv)
print(tvminustau)
print (norm(input_array))

#sets the constraint that tv(model) <= tau
cons = ({'type': 'ineq', 'fun': total_variation2d_inequality, 'args': (param, ) })

#res = optimize.minimize(norm, input_array, tol = 1e-6)
res = optimize.minimize(norm, input_array, tol = 1e-6, constraints = cons)
print(res.x)
print(res.success)
print(res.message)
print(res.fun)
print(total_variation2d_inequality(res.x, param) )



