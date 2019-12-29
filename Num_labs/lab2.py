import numpy as np

import solvers


def first_equation(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)

	u[0] = -1.3e-3 * u2 - 1e3 * u1 * u2 - 2.5e3 * u1 * u3
	u[1] = -1.3e-3 * u2 - 1e3 * u1 * u2
	u[2] =                              - 2.5e3 * u1 * u3

	return u

def fe_jacob(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = -1e3 * u2 - 2.5e3 * u3; res[0][1] = -1.3e-3 * u2 - 1e3 * u1; res[0][2] = -2.5e3 * u1;
	res[1][0] = -1e3 * u2;              res[1][1] = -1.3e-3 * u2 - 1e3 * u1; res[1][2] = 0.0;
	res[2][0] =           - 2.5e3 * u3; res[2][1] = 0.0;                     res[2][2] = -2.5e3 * u1;
	
	return res

def solve_first_equation_rki():
	t0 = 0.0
	u0 = np.float64((0, 1, 1))
	
	n = 10000
	tn = 500.0

	dt = tn / n

	solver = solvers.RKI_naive(
		  solvers.lobattoIIIC_4()
		, first_equation
		, fe_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, u0
	)

	for i in range(n):
		if i % 100 == 0:
			u1, u2, u3 = solver.value()
			print(u1, ' ', u2, ' ', u3, '---', u1 - u2 - u3)
		solver.evolve(t0 + i * dt, dt)
	print(solver.value())

def solve_first_equation():
	t0 = 0.0
	u0 = np.float64((0, 1, 1))
	
	n = 5000000
	tn = 500.0

	dt = tn / n


	order = 4 # order, change this


	initial_solver = solvers.RKI_naive(
		  solvers.lobattoIIIC_4()
		, first_equation
		, fe_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, u0
	)

	solver = solvers.AdamsImplicitSolver(
		  order 
		, initial_solver
		, first_equation
		, fe_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, t0, dt
	)
	
	#     ->    ->    ->              ->   ->      ->
	# t(0)  t(1)  t(2)  ... t(order-1)  ...  t(n-1)  t(n)
	for i in range(order - 1, n):
		if i % 100 == 0:
			u1, u2, u3 = solver.value()
			print(u1, ' ', u2, ' ', u3, '---', u1 - u2 - u3)
		solver.evolve(t0 + i * dt, dt)
	print(solver.value())
	

def second_equation(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)

	u[0] = -0.04 * u1 + 1e4 * u1 * u3
	u[1] = +0.04 * u1 - 1e4 * u1 * u3 - 3e7 * u2 * u2
	u[2] =                            + 3e7 * u2 * u2
	
	return u

def se_jacob(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = -0.04 + 1e4 * u3; res[0][1] = 0;             res[0][2] = +1e4 * u1;
	res[1][0] = +0.04 - 1e4 * u3; res[1][1] = -3e7 * 2 * u2; res[1][2] = -1e4 * u1;
	res[2][0] = 0;                res[2][1] = +3e7 * 2 * u2; res[2][2] = 0;
	
	return res

def solve_second_equation_rki():
	t0 = 0.0
	u0 = np.float64((1.0, 0.0, 1.0))
	
	n = 4000
	tn = 40.0

	dt = tn / n

	solver = solvers.RKI_naive(
		  solvers.lobattoIIIC_2()
		, second_equation
		, se_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, u0
	)

	for i in range(n):
		if i % 100 == 0:
			u1, u2, u3 = solver.value()
			print(u1, ' ', u2, ' ', u3, '---', u1 + u2 + u3)
		solver.evolve(t0 + i * dt, dt)
	print(solver.value())

def solve_second_equation():
	t0 = 0.0
	u0 = np.float64((0.1, 0.3, 0.4))
	
	n = 1000000000
	tn = 10.0

	dt = tn / n


	order = 4 # order, change this


	initial_solver = solvers.RKI_naive(
		  solvers.gauss_legendre_6()
		, second_equation
		, se_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, u0
	)

	solver = solvers.AdamsImplicitSolver(
		  order 
		, initial_solver
		, second_equation
		, se_jacob
		, solvers.NeutonSolver(1e-15, 200)
		, t0, dt
	)
	
	#     ->    ->    ->              ->   ->      ->
	# t(0)  t(1)  t(2)  ... t(order-1)  ...  t(n-1)  t(n)
	for i in range(order - 1, n):
		if i % 100 == 0:
			u = solver.value()
			print(u, ' ', t0 + i * dt, ' ', u[0] + u[1] + u[2])
		solver.evolve(t0 + i * dt, dt)

	u = solver.value()
	print('Value: ', u[0], ' ', u[1], ' ', u[2])
	print('Invariant: ', u[0] + u[1] + u[2])


def main():
	print('----Testing first equation----')
	solve_first_equation()

	#print('----Testing second equation----')
	#solve_first_equation_rki()

	#print('----Testing second equation----')
	#solve_second_equation()

	#print('----Testing second equation----')
	#solve_second_equation_rki()


if __name__ == '__main__':
	main()
	print('God bless python. Press any key...', end='')
	input()
	print('God bless python. Press any key...', end='')
	input()
	print('God bless python. Press any key...', end='')
	input()