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
	

def second_equation(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)

	u[0] = -0.04 * u1 + 1e4 * u2 * u3
	u[1] = +0.04 * u1 - 1e4 * u2 * u3 - 3e7 * u2 * u2
	u[2] =                            + 3e7 * u2 * u2
	
	return u

def se_jacob(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = -0.04 + 1e4 * u3; res[0][1] = +1e4 * u3               ; res[0][2] = +1e4 * u2;
	res[1][0] = +0.04 - 1e4 * u3; res[1][1] = -1e4 * u3 - 3e7 * 2 * u2; res[1][2] = -1e4 * u2;
	res[2][0] = 0;                res[2][1] =           + 3e7 * 2 * u2; res[2][2] = 0;
	
	return res


def integrate_equation(solver, dt, tn, output_every=1000):
	i = 0
	while True:
		if i % output_every == 0:
			t, u = solver.get_state()
			print(f't: {t} | u: {u} |')
		i += 1

		t = solver.t
		if t < tn:
			solver.evolve(t, dt)
		else:
			break

def solve_first():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((0, 1, 1))                          # initial value
	tn = 500.0                                          # final time
	un = np.float64((-1.893e-7, 0.5976547, 1.40223434)) # final value(for test)
	n = 5000                                            # count of intervals
	dt = (tn - t0) / n                                  # time delta

	f = first_equation
	j = fe_jacob

	# Initialize your solver here and call integrate_equation()
	solver = solvers.RKI_naive(solvers.lobattoIIIC_2(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)

	# ini_solver = solvers.RKI_naive(solvers.gauss_legendre_6(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)

	# a, b = solvers.build_implicit_adams(2)
	# solver = solvers.ImplicitMultistepSolver(ini_solver, solvers.NeutonSolver(1e-15, 100), a, b, dt)

	integrate_equation(solver, dt, tn)

	yn = solver.value()

	# Results
	print('---First equation. Expected and got result---')
	print(f'Expected:{un}')
	print(f'Got     :{yn}')

def solve_second():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((1, 0, 0))                          # initial value
	tn = 40.0                                           # final time
	un = np.float64((0.7158271, 9.186e-6, 0.2841637))   # final value(for test)
	n = 80000000                                          # count of intervals
	dt = (tn - t0) / n                                  # time delta

	f = second_equation
	j = se_jacob

	# Initialize your solver here and call integrate_equation()
	# solver = solvers.RKI_naive(solvers.gauss_legendre_6(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)
	solver = solvers.RKE(solvers.classic_4(), f, t0, u0)

	integrate_equation(solver, dt, tn, 100000)

	yn = solver.value()

	# Results
	print('---Second equation. Expected and got result---')
	print(f'Expected:{un}')
	print(f'Got     :{yn}')


def main():
	#solve_first()
	solve_second()


if __name__ == '__main__':
	main()

	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()