import numpy as np
import math  as m
import cmath as cm
import copy  as c
import matplotlib.pyplot as plt
import solvers


class IdentityFunction:
	def __call__(self, x):
		return x

class ConstantFunction:

	def __init__(self, const):
		self.const = const
	
	def __call__(self, x):
		return self.const

class MultiplyFunction:
	def __init__(self, left_callable, right_callable):
		self.left_callable  = left_callable
		self.right_callable = right_callable

	def __call__(self, x):
		left  = self.left_callable(x)
		right = self.right_callable(x)
		return  left * right

class DivideFunction:

	def __init__(self, num, denom):
		self.num   = num
		self.denom = denom

	def __call__(self, arg):
		num   = self.num(arg)
		denom = self.denom(arg)
		return num / denom

class ExpFunction:

	def __init__(self, callable):
		self.callable = callable

	def __call__(self, arg):
		exp = np.exp(self.callable(arg))
		return exp

class PolynomialFunction:
	
	def __init__(self, coefs, callable):
		self.callable   = callable
		self.polynomial = np.polynomial.Polynomial(coefs)

	def __call__(self, arg):
		poly = self.polynomial(self.callable(arg))
		return poly



class Method:
	def __init__(self, alpha_coefs, beta_coefs):
		id    = IdentityFunction()
		const = ConstantFunction(1j)
		mul   = MultiplyFunction(const, id)
		exp   = ExpFunction(mul)

		alpha_poly = PolynomialFunction(alpha_coefs, exp)
		beta_poly  = PolynomialFunction(beta_coefs , exp)

		frac = DivideFunction(alpha_poly, beta_poly)

		self.callable = frac
		self.alpha_poly = alpha_poly.polynomial
		self.beta_poly  = beta_poly.polynomial

	def __call__(self, arg):
		return self.callable(arg)



def get_scatter_points(method, x0, x1, y0, y1, split_x, split_y):
	z = []

	dx = (x1 - x0) / split_x
	dy = (y1 - y0) / split_y

	for i in range(split_x + 1):
		for j in range(split_y + 1):
			z_ij = np.complex(x0 + i * dx, y0 + j * dy)
			
			poly = method.alpha_poly - method.beta_poly * z_ij
			
			all_less = True
			for root in poly.roots():
				if abs(root) > 1.0:
					all_less = False
					break
			if all_less:
				z.append(z_ij)

	return z
						
def compute_points(func, a, b, split):
	step = (b - a) / split
	points = []
	for i in range(split + 1):
		z = func(a + i * step)
		points.append(z)
	points.append(points[0])
	return points		

def split_z(complex_seq):
	x = []
	y = []
	for z in complex_seq:
		x.append(z.real)
		y.append(z.imag)
	return x, y

def parse_coefs(coefs_str):

	return [np.float64(coef) for coef in coefs_str.split()]

def plot_method(method):
	z_bound = compute_points(method, -m.pi + 1e-5, m.pi - 1e-5, 300)
	x_bound, y_bound = split_z(z_bound)

	x_min, x_max = 1.1 * min(x_bound) - 0.5, 1.1 * max(x_bound) + 0.5
	y_min, y_max = 1.1 * min(y_bound) - 0.5, 1.1 * max(y_bound) + 0.5
 
	z_domain = get_scatter_points(method, x_min, x_max, y_min, y_max, 50, 50)
	x_domain, y_domain = split_z(z_domain)

	plt.scatter(x_domain, y_domain, [0.3 for z in z_domain])
	plt.plot(x_bound, y_bound)

	plt.show()



def demo_explicit():
	print('---Explicit methods---')
	methods = {
		  'Adams–Bashforth1' : {'alpha' : [-1, 1], 'beta' : [1]}
		, 'Adams–Bashforth2' : {'alpha' : [0, -1, 1], 'beta' : [-1/2, 3/2]}
		, 'Adams–Bashforth3' : {'alpha' : [0, 0, -1, 1], 'beta' : [5/12, -16/12, 23/12]}
		, 'Adams–Bashforth4' : {'alpha' : [0, 0, 0, -1, 1], 'beta' : [-9/24, 37/24, -59/24, 55/24]}
		, 'Adams–Bashforth5' : {'alpha' : [0, 0, 0, 0, -1, 1], 'beta' : [251/720, -1274/720, 2616/720, -2774/720, 1901/720]}
	}

	for name, coefs in methods.items():
		print('Plotting domain of stability of the ', name)
		method = Method(coefs['alpha'], coefs['beta'])
		plot_method(method)

def demo_implicit():
	print('---Implicit methods---')
	methods = {
		  'Adams–Moulton1' : {'alpha' : [-1, 1], 'beta' : [0, 1]}
		, 'Adams–Moulton2' : {'alpha' : [-1, 1], 'beta' : [1/2, 1/2]}
		, 'Adams–Moulton3' : {'alpha' : [0, -1, 1], 'beta' : [-1/12, 2/3, 5/12]}
		, 'Adams–Moulton4' : {'alpha' : [0, 0, -1, 1], 'beta' : [1/24, -5/24, 19/24, 9/24]}
		, 'Adams–Moulton5' : {'alpha' : [0, 0, 0, -1, 1], 'beta' : [-19/720, 106/720, -264/720, 646/720, 251/720]} 
	}

	for name, coefs in methods.items():
		print('Plotting domain of stability of the ', name)
		method = Method(coefs['alpha'], coefs['beta'])
		plot_method(method)

def demo():
	# EXAMPLE
	a,b = solvers.build_bdf_6()
	plot_method(Method([0, -1, +1], [-1/6, 5/6, 1/3]))


if __name__ == '__main__':
	demo()
