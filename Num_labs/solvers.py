import math as m
import numpy as np
import numpy.polynomial as poly
import numpy.linalg as lin
import linsolve


#utility
def factorial(num):
	res = 1
	while num > 1:
		res *= num
		num -= 1
	return res

def kronecker_delta(i, j):

	return 1 if i == j else 0

			
class DiffJacobian1:
	def __init__(self, func, eps = 1e-6):
		self.func = func
		self.eps  = eps

	def __call__(self, u):
		order = len(u)
		jac = np.zeros((order, order), np.float64)

		um1 = u.copy()
		up1 = u.copy()
		#iterate through variables
		for j in range(order):
			#assign u
			for i in range(order):
				um1[i] = u[i]
			for i in range(order):
				up1[i] = u[i]

			#add/subtract eps from i-th component
			um1[j] -= self.eps
			up1[j] += self.eps

			fm1 = self.func(um1)
			fp1 = self.func(up1)

			#fill j-th column
			for i in range(order):
				jac[i][j] = (fp1[i] - fm1[i]) / (2 * self.eps)

		return jac

class DiffJacobian2:
	def __init__(self, func, eps = 1e-6):
		self.func = func
		self.eps  = eps

	def __call__(self, t, u):
		order = len(u)
		jac = np.zeros((order, order), np.float64)

		um1 = u.copy()
		up1 = u.copy()
		#iterate through variables
		for j in range(order):
			#assign u
			for i in range(order):
				um1[i] = u[i]
			for i in range(order):
				up1[i] = u[i]

			#add/subtract eps from i-th component
			um1[j] -= self.eps
			up1[j] += self.eps

			fm1 = self.func(t, um1)
			fp1 = self.func(t, up1)

			#fill j-th column
			for i in range(order):
				jac[i][j] = (fp1[i] - fm1[i]) / (2 * self.eps)

		return jac

class BindFirst:
	def __init__(self, func, first):
		self.func  = func
		self.first = first

	def __call__(self, second):
		return self.func(self.first, second)

class LinearCombination:
	def __init__(self, func, A, B):
		self.func = func
		self.A = A
		self.B = B

	def __call_(self, arg):
		return A * self.func(arg) + B



#solver for nonlinear systems
class NeutonSolver:
	def __init__(self, eps, iter_lim):
		self.eps = eps
		self.iter_lim = iter_lim

	def solve(self, func, jacob, x0):
		x_k = x0.copy()
		iters = 0

		while True:
			sys  = jacob(x_k)
			term = -func(x_k)

			delta = lin.solve(sys, term)
			#delta = linsolve.gaussian_elimination_solve(sys, term)

			x_k += delta
			iters += 1

			if abs(delta).max() < self.eps or iters >= self.iter_lim:
				break

		return x_k


# solver base class
class SolverBase:
	def __init__(self, function, jacobian, t0, u0):
		self.function = function
		self.jacobian = jacobian

		self.t = t0
		self.u = u0

	def set_state(self, t, u):
		self.t = t
		self.u = u

	def get_state(self):
		return self.t, self.u

	def value(self):
		return self.u
	
	def evolve(self, t, dt):
		# to override
		pass
	

#Runge-Kutta methods
class Tableau:
	def __init__(self, order, aMat, bVec, cVec):
		self.order = order
		self.aMat = np.copy(aMat)
		self.bVec = np.copy(bVec)
		self.cVec = np.copy(cVec)

#explicit
def classic_4():
	aMat = np.float64(
		(
			  (0.0, 0.0, 0.0, 0.0)
			, (0.5, 0.0, 0.0, 0.0)
			, (0.0, 0.5, 0.0, 0.0)
			, (0.0, 0.0, 1.0, 0.0)
		)
	)

	bVec = np.float64((1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0))

	cVec = np.float64((0.0, 0.5, 0.5, 1.0))

	return Tableau(4, aMat, bVec, cVec)

#implicit
def backward_euler_1():
	a_mat = np.float64( ((1,),) )
	b_vec = np.float64( (1,) )
	c_vec = np.float64( (1,) )

	return Tableau(1, a_mat, b_vec, c_vec)

def implicit_midpoint_2():
	a_mat = np.float64(((0.5,),))
	b_vec = np.float64((1.0,))
	c_vec = np.float64((0.5,))

	return Tableau(1, a_mat, b_vec, c_vec)

def kra_spi_2():
	a_mat = np.float64((
		  (+1/2, 0)
		, (-1/2, 2) 
	))
	b_vec = np.float64((-1/2, 3/2))
	c_vec = np.float64((+1/2, 3/2))

	return Tableau(2, a_mat, b_vec, c_vec)

def qin_zha_2():
	a_mat = np.float64((
		  (+1/4, 0)
		, (+1/2, 1/4) 
	))
	b_vec = np.float64((1/2, 1/2))
	c_vec = np.float64((1/4, 3/4))

	return Tableau(2, a_mat, b_vec, c_vec)

def lobattoIIIC_2():
	a_mat = np.float64((
		  (1/2, -1/2)
		, (1/2,  1/2) 
	))
	b_vec = np.float64((1/2, 1/2))
	c_vec = np.float64(( 0 ,  1 ))

	return Tableau(2, a_mat, b_vec, c_vec)

def lobattoIIIC_4():
	a_mat = np.float64((
		  (1/6, -1/3,  1/6 )
		, (1/6, 5/12, -1/12) 
		, (1/6,  2/3,  1/6 )
	))
	b_vec = np.float64((1/6, 2/3, 1/6))
	c_vec = np.float64((0  , 1/2,   1))

	return Tableau(3, a_mat, b_vec, c_vec)

def gauss_legendre_6():
	sq15 = m.sqrt(15.0)
	
	a_mat = np.float64((
		  (5/36          , 2/9 - sq15/15, 5/36 - sq15/30)
		, (5/36 + sq15/24, 2/9          , 5/36 - sq15/24) 
		, (5/36 + sq15/30, 2/9 + sq15/15, 5/36          )
	))
	b_vec = np.float64((5/18, 4/9, 5/18))
	c_vec = np.float64((1/2 - sq15/10, 1/2, 1/2 + sq15/10))

	return Tableau(3, a_mat, b_vec, c_vec)


class RKE:
	def __init__(self, tableau, func, u0):
		self.tableau = tableau
		self.rke_order = tableau.order
		self.sys_order = len(u0)
		self.u = u0.copy()

		self.f = func

	def value(self):

		return self.u

	def evolve(self, t, dt):
		aMat, bVec, cVec = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		kVecs = np.zeros((self.rke_order, self.sys_order), np.float64)
		for i in range(self.rke_order):
			for j in range(i):
				kVecs[i] += dt * kVecs[j] * aMat[i][j]
			kVecs[i] = self.f(t + dt * cVec[i], self.u + kVecs[i])

		du = np.zeros((self.sys_order, ), np.float64)
		for i in range(self.rke_order):
			du += dt * bVec[i] * kVecs[i]

		self.u += du

# TODO : specialJacobian -> inner class SpecialJacobian
# TODO : specialFunction -> inner class SpecialFunction
class RKI_naive:
	def __init__(self, tableau, func, jacob, neuton_solver, u0):
		self.tableau = tableau
		self.rki_order = tableau.order
		self.sys_order = len(u0)

		self.u  = u0.copy()
		self.t  = 0.0
		self.dt = 0.0

		self.function = func
		self.jacobian = jacob
		self.neuton_solver = neuton_solver	

	def value(self):

		return self.u

	def evolve(self, t, dt):
		self.t  = t
		self.dt = dt

		u = self.u

		N = self.sys_order
		s = self.rki_order

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec
		
		#initial guess
		value = self.function(t, u)
		z = np.zeros((s, N), np.float64)
		for i in range(s):
			z[i] = value
		z = z.reshape((N * s))

		#solving system
		z = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, z)

		#computing delta
		z = z.reshape((s, N))

		du = np.zeros((N,), np.float64)
		for i in range(s):
			du += b[i] * z[i]
		du *= dt

		#update result
		self.u += du

	def __special_jacobian(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_jacobian = np.zeros((N * s, N * s), np.float64)
		for i in range(s):
			sums = np.zeros((N,), np.float64)
			for j in range(s):
				sums += a[i][j] * z[j]
			sums *= dt

			jacobian = self.jacobian(t + dt * c[i], u + sums)
			for I in range(N):
				for j in range(s):
					for J in range(N):
						dij = kronecker_delta(i, j)
						dIJ = kronecker_delta(I, J)

						elem = dij * dIJ - dt * a[i][j] * jacobian[I][J]

						special_jacobian[N * i + I][N * j + J] = elem

		return special_jacobian

	def __special_function(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))
		
		special_function = np.zeros((s, N), np.float64)
		for i in range(s):
			sums = np.zeros((N,), np.float64)
			for j in range(s):
				sums += a[i][j] * z[j]
			sums *= dt

			special_function[i] = self.function(t + dt * c[i], u + sums)
				
		z = z.reshape((N * s))
		special_function = special_function.reshape((N * s))

		return z - special_function		

# TODO : specialJacobian -> inner class SpecialJacobian
# TODO : specialFunction -> inner class SpecialFunction		
class RKI_better:
	def __init__(self, tableau, func, jacob, neuton_solver, u0):
		self.tableau = tableau
		self.rki_order = tableau.order
		self.sys_order = len(u0)

		self.u  = u0.copy()
		self.t  = 0.0
		self.dt = 0.0

		self.function = func
		self.jacobian = jacob
		self.neuton_solver = neuton_solver	

	def value(self):

		return self.u

	def evolve(self, t, dt):
		self.t  = t
		self.dt = dt

		u = self.u

		N = self.sys_order
		s = self.rki_order

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec
		
		#initial guess
		z = np.zeros((N * s), np.float64)

		#solving system
		z = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, z)

		#computing delta
		z = z.reshape((s, N))
		
		du = np.zeros((N,), np.float64)
		for i in range(s):
			du += b[i] * self.function(t + dt * c[i], u + z[i])
		du *= dt

		#update result
		self.u += du

	def __special_jacobian(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_jacobian = np.zeros((s, N, s, N), np.float64)
		for j in range(s):
			jacobian = self.jacobian(t + c[j] * dt, u + z[j])

			for i in range(s):
				dij = kronecker_delta(i, j)

				for I in range(N):

					for J in range(N):
						dIJ = kronecker_delta(I, J)

						special_jacobian[i][I][j][J] = dij * dIJ - dt * a[i][j] * jacobian[I][J]

		return special_jacobian.reshape((N * s, N * s))

	def __special_function(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_function = np.zeros((s, N), np.float64)
		special_function[:,:] = z[:,:]
		
		for j in range(s):
			function = self.function(t + c[j] * dt, u + z[j])
			for i in range(s):
				special_function[i] -= dt * a[i][j] * function

		return special_function.reshape((N * s))



#coefs for Adams methods
def build_explicit_adams(order):
	# produces list of order elements
	# coefs go from lower degree to the higher

	s = order

	if s == 1:
		return [1]

	b_coefs = []
	for j in range(s):
		all_roots = [-i for i in range(s) if i != j]

		integrand = poly.Polynomial.fromroots(all_roots)
		integral  = integrand.integ()			
		value     = integral(1.0)		

		minus_one = 1 if j % 2 == 0 else -1
		coef      = minus_one / factorial(j) / factorial(s - j - 1) * value

		b_coefs.append(coef)

	b_coefs.reverse()
	return b_coefs

def build_implicit_adams(order):
	# produces list of orer + 1 elements
	# coefs go from lower degree to the higher

	s = order

	if s == 0:
		return [0, 1]

	b_coefs = []
	for j in range(s + 1):
		all_roots = [-(i - 1) for i in range(s + 1) if i != j]

		integrand = poly.Polynomial.fromroots(all_roots)
		integral  = integrand.integ()			
		value     = integral(1.0)		

		minus_one = 1 if j % 2 == 0 else -1
		coef      = minus_one / factorial(j) / factorial(s - j) * value

		b_coefs.append(coef)

	b_coefs.reverse()
	return b_coefs


# TODO : specialJacobian -> inner class SpecialJacobian
# TODO : specialFunction -> inner class SpecialFunction
class AdamsExplicitSolver:
	def __init__(self, order, ivp_solver, function, t, dt):
		# order >= 1
		# ivp_solver should guarantee appropriate accuracy

		self.function = function

		self.sys_order = len(ivp_solver.value())

		self.dt = dt

		self.coefs = np.float64(build_explicit_adams(order))
		self.steps = len(self.coefs)

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()[:]
			ivp_solver.evolve(t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()[:]

	def value(self):

		return self.values[-1]

	def evolve(self, t, dt):
		# dt - ignored

		f  = self.function
		y  = self.values
		a  = self.coefs
		dt = self.dt

		next = y[-1].copy()
		for i in range(self.steps):
			j = (self.steps - 1) - i
			next += dt * a[i] * f(t - j * dt, y[i])

		self.__right_round_like_a_record_baby(next)

	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1][:]
		self.values[-1][:] = value[:]

# TODO : specialJacobian -> inner class SpecialJacobian
# TODO : specialFunction -> inner class SpecialFunction
class AdamsImplicitSolver:
	def __init__(self, order, ivp_solver, function, jacobian, neuton_solver, t, dt):
		# order >= 0
		# ivp_solver should guarantee appropriate accuracy

		self.function = function
		self.jacobian = jacobian
		self.neuton_solver = neuton_solver

		self.sys_order = len(ivp_solver.value())

		self.t  = None
		self.dt = dt
		self.A  = None
		self.B  = None

		self.coefs = np.float64(build_implicit_adams(order))
		self.steps = len(self.coefs) - 1

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()
			ivp_solver.evolve(t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()

	def value(self):

		return self.values[-1]

	def evolve(self, t, dt):
		# dt - ignored

		f = self.function
		y = self.values
		a = self.coefs
		dt = self.dt

		B = y[-1].copy()
		for i in range(self.steps):
			j = (self.steps - 1) - i
			B += dt * a[i] * f(t - j * dt, y[i])
		
		A = dt * a[-1]

		self.t = t
		self.A = A
		self.B = B

		next = y[-1].copy()
		next = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, next)

		self.__right_round_like_a_record_baby(next)

	def __special_function(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt

		f = self.function

		return y - A * f(t + dt, y) - B
		
	def __special_jacobian(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt

		j = self.jacobian

		special_jacobian = -A * j(t + dt, y)
		for i in range(self.sys_order):			
			special_jacobian[i][i] += 1.0
		return special_jacobian
	
	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1]
		self.values[-1][:] = value

# TODO
class ExplicitMultiStepSolver:
	pass

# TODO
class ImplicitMultiStepSolver:
	pass


# TODO
#solver with automatic step selection
class AutoSolver:
	def __init__(self, ivp_solver, eps):
		self.ivp_solver = ivp_solver
		self.eps = abs(eps)

	def value(self):
		
		return self.ivp_solver.value()

	def evolve(self, t, dt):
		while True:
			while True:
				break
			break


#tests
def test_adams_coefs():
	print(build_explicit_adams(4))
	print(build_implicit_adams(4))


def test_equation(t, u):
	v, w = u[0], u[1]

	res = np.zeros((2,), np.float64)
	res[0] = w
	res[1] = m.exp(t) - 2 * w - v

	return res

def test_equation1(t, u):
	res = np.zeros((1,), np.float64)
	res[0] = 2 * t * u
	return res

def solution(t):
	em = m.exp(-t)
	ep = m.exp(+t)

	return np.float64((em * t + ep / 4, em * (-t + 1) + ep / 4))

def solution1(t):
	res = np.zeros((1,), np.float64)
	res[0] = m.exp(t * t) 
	return res

def test_solver_value(solver, t0, dt, count, test):
	for i in range(count):
		solver.evolve(t0 + i * dt, dt)
	print(solver.value(), ' ', test)
	return solver.value() - test

def get_values(solver, t0, dt, count):
	values = []
	for i in range(count):
		values.append(solver.value().copy())
		solver.evolve(t0 + i * dt, dt)
	values.append(solver.value().copy())

	return values

def test_rk_solvers():
	equ = test_equation
	sol = solution

	t0 = 0.0
	u0 = sol(t0)
	count = 1000
	dt = 10.0 / count

	test_val = sol(dt * count)


	rke_solver = RKE(classic_4(), equ, u0)
	rki_solver0 = RKI_better(qin_zha_2(), equ, DiffJacobian2(equ), NeutonSolver(1e-15, 70), u0)
	rki_solver1 = RKI_better(kra_spi_2(), equ, DiffJacobian2(equ), NeutonSolver(1e-15, 70), u0)
	rki_solver2 = RKI_better(gauss_legendre_6(), equ, DiffJacobian2(equ), NeutonSolver(1e-15, 70), u0)

	print(test_solver_value(rke_solver , t0, dt, count, test_val))
	print(test_solver_value(rki_solver0, t0, dt, count, test_val))
	print(test_solver_value(rki_solver1, t0, dt, count, test_val))
	print(test_solver_value(rki_solver2, t0, dt, count, test_val))

def test_adams_solvers():
	t0 = 0.0
	u0 = solution(t0)
	count = 100000
	dt = 1.0 / count

	test_val = solution(1.0)

	rki_solver0 = RKI_better(lobattoIIIC_4(), test_equation, DiffJacobian2(test_equation), NeutonSolver(1e-15, 50), u0)
	rki_solver1 = RKI_better(lobattoIIIC_4(), test_equation, DiffJacobian2(test_equation), NeutonSolver(1e-15, 50), u0)
	rk_solver   = RKI_naive(gauss_legendre_6(), test_equation, DiffJacobian2(test_equation), NeutonSolver(1e-15, 50), u0)

	order = 4
	ae_solver = AdamsExplicitSolver(order, rki_solver0, test_equation, t0, dt)
	ai_solver = AdamsImplicitSolver(order, rki_solver1, test_equation, DiffJacobian2(test_equation), NeutonSolver(1e-15, 50), t0, dt)

	#ae_values = get_values(ae_solver, t0 + (order - 1) * dt, dt, count - order)
	#ai_values = get_values(ai_solver, t0 + (order - 1) * dt, dt, count - order)
	#rk_values = get_values(rk_solver, t0                   , dt, count        )[order - 1:]
	
	#for i in range(len(ae_values)):
	#	print(ae_values[i], ' ', ai_values[i], ' ', rk_values[i])

	print(test_solver_value(ae_solver, t0 + (order - 1) * dt, dt, count - order + 1, test_val))
	print(test_solver_value(ai_solver, t0 + (order - 1) * dt, dt, count - order + 1, test_val))
	print(test_solver_value(rk_solver, t0, dt, count, test_val))

def test_auto_solver():

	pass

if __name__ == '__main__':
	test_rk_solvers()
	#test_adams_coefs()