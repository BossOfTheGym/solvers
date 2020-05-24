import numpy as np
import numpy.linalg as lin
import math as m

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

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


def third_equation(t, u):
	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)
	u[0] = a * u2 * u3
	u[1] = b * u3 * u1
	u[2] = c * u1 * u2	
	return u

def te_jacob(t, u):
	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = 0;      res[0][1] = a * u3; res[0][2] = a * u2;
	res[1][0] = b * u3; res[1][1] = 0;      res[1][2] = b * u1;
	res[2][0] = c * u2; res[2][1] = c * u1; res[2][2] = 0;
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
	n = 80000000                                        # count of intervals
	dt = (tn - t0) / n                                  # time delta

	f = second_equation
	j = se_jacob

	# Initialize your solver here and call integrate_equation()
	# solver = solvers.RKI_naive(solvers.gauss_legendre_6(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)
	solver = solvers.RKE(solvers.classic_4(), f, t0, u0)

	integrate_equation(solver, dt, tn)

	yn = solver.value()

	# Results
	print('---Second equation. Expected and got result---')
	print(f'Expected:{un}')
	print(f'Got     :{yn}')

def solve_third():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((m.cos(1.1), 0, m.sin(1.1)))        # initial value
	tn = 100.0                                          # final time
	n = 1000										    # count of intervals
	dt = (tn - t0) / n                                  # time delta
	output_every = 1

	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u01, u02, u03 = u0
	i1 = u01 * u01 + u02 * u02 + u03 * u03
	i2 = u01 * u01 / I1 + u02 * u02 / I2 + u03 * u03 / I3

	f = third_equation
	j = te_jacob

	def integrate_with_solver(solver):
		i = 0
		t_values = []
		u1_values = []
		u2_values = []
		u3_values = []
		inv_first  = []
		inv_second = []
		while True:
			if i % output_every == 0:
				t, u = solver.get_state()
				u1, u2, u3 = u

				t_values.append(t)
				u1_values.append(u1)
				u2_values.append(u2)
				u3_values.append(u3)

				inv_first.append(u1 * u1 + u2 * u2 + u3 * u3)
				inv_second.append(u1 * u1 / I1 + u2 * u2 / I2 + u3 * u3 / I3)
			i += 1

			t = solver.t
			if t < tn:
				solver.evolve(t, dt)
			else:
				break

		max_delta = 0.0
		for inv in inv_first:
			max_delta = max(max_delta, m.fabs(i1 - inv))
		print('Maximum first invariant deviation: ', max_delta)

		max_delta = 0.0
		for inv in inv_second:
			max_delta = max(max_delta, m.fabs(i2 - inv))
		print('Maximum second invariant deviation: ', max_delta)

		fig = pyplot.figure()
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.scatter(u1_values, u2_values, u3_values);
		pyplot.show()

	solver = solvers.RKI_naive(solvers.implicit_midpoint_2(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)
	integrate_with_solver(solver)

	solver = solvers.RKE(solvers.classic_4(), f, t0, u0)
	integrate_with_solver(solver)
	
	solver = solvers.RKE(solvers.explicit_euler_1(), f, t0, u0)
	integrate_with_solver(solver)
	

def kepler_test():	
	k = 0.005

	def kepler_H(p, q):
		pp = np.dot(p, p)
		qq = np.dot(q, q)

		qSqR = np.sqrt(qq)

		res = np.zeros((1, 1), np.float64)
		res[0,0] = pp / 2 - 1 / qSqR - k / (qSqR * qq)
		return res

	def kepler_dHdp(p, q):
		
		return np.float64(p)

	def kepler_dHdq(p, q):
		pp = np.dot(p, p)
		qq = np.dot(q, q)

		qSqR = np.sqrt(qq)

		return -(3 * k - qSqR) / (qSqR * qq) * (q / qSqR)

	def kepler_dH(p, q):
		newP = np.zeros((2,), np.float64)
		newQ = np.zeros((2,), np.float64)
		newP = kepler_dHdp(p, q)
		newQ = kepler_dHdq(p, q)
		return newP, newQ


	def kepler_L(p, q):
		p1, p2 = p[0], p[1]
		q1, q2 = q[0], q[1]

		res = np.zeros((1, 1), np.float64)
		res[0, 0] = q1 * p2 - q2 * p1
		return res

	def kepler_dL(p, q):
		p1, p2 = p[0], p[1]
		q1, q2 = q[0], q[1]

		newP = np.zeros((2,), np.float64)
		newQ = np.zeros((2,), np.float64)
		newP[0] = -q2
		newP[1] =  q1
		newQ[0] =  p2
		newQ[1] = -p1
		return newP, newQ


	def kepler_equation(p, q):

		return (-kepler_dHdq(p, q), kepler_dHdp(p, q))

	def kepler_adaptor(t, u):
		p = u[0 : 2]
		q = u[2 : 4]
		p, q = kepler_equation(p, q)
		res = np.zeros((4,), np.float64)
		res[0 : 2] = p
		res[2 : 4] = q
		return res

	
	def project_to_H(p, q, H0):
		# to matrix
		H0 = np.float64(H0).reshape((1, 1))

		# vector to project
		y = np.zeros((4,), np.float64)
		y[0 : 2] = p
		y[2 : 4] = q

		# derivatives of jacobian
		dp, dq = kepler_dH(p, q)

		dgT = np.zeros((4, 1), np.float64)
		dgT[0 : 2, :] = dp.reshape((2, 1))
		dgT[2 : 4, :] = dq.reshape((2, 1))
		dg = dgT.reshape((1, 4))

		# inverse of matrix
		dgdgTi = lin.inv(dg @ dgT)

		# lambda vec		
		l = np.zeros((1,), np.float64)

		# first simplified neuton iteration
		tmp = y + dgT @ l
		p = tmp[0 : 2]
		q = tmp[2 : 4]
		l += -(dgdgTi @ (kepler_H(p, q) - H0)).reshape((1,))

		# second simplified neuton iteration
		tmp = y + dgT @ l
		p = tmp[0 : 2]
		q = tmp[2 : 4]
		l += -(dgdgTi @ (kepler_H(p, q) - H0)).reshape((1,))

		ret = y + dgT @ l
		return ret[0 : 2], ret[2 : 4]
			
		
	def project_to_HL(p, q, H0, L0):
		# to matrix
		H0 = np.float64(H0).reshape((1, 1))
		L0 = np.float64(L0).reshape((1, 1))

		# vector to project
		y = np.zeros((4,), np.float64)
		y[0 : 2] = p
		y[2 : 4] = q

		# derivatives of jacobian
		dp0, dq0 = kepler_dH(p, q)
		dp1, dq1 = kepler_dL(p, q)

		dgT = np.zeros((4, 2), np.float64)
		dgT[0 : 2, 0] = dp0
		dgT[2 : 4, 0] = dq0
		dgT[0 : 2, 1] = dp1
		dgT[2 : 4, 1] = dq1
		dg = np.transpose(dgT)

		# inverse of matrix
		dgdgTi = lin.inv(dg @ dgT)

		# lambda vec		
		l = np.zeros((2,), np.float64)

		# first simplified neuton iteration
		tmp = y + dgT @ l
		p = tmp[0 : 2]
		q = tmp[2 : 4]
		tmp = np.zeros((2,), np.float64)
		tmp[0] = (kepler_H(p, q) - H0)[0, 0]
		tmp[1] = (kepler_L(p, q) - L0)[0, 0]
		l += -dgdgTi @ tmp

		# second simplified neuton iteration
		tmp = y + dgT @ l
		p = tmp[0 : 2]
		q = tmp[2 : 4]
		tmp = np.zeros((2,), np.float64)
		tmp[0] = (kepler_H(p, q) - H0)[0, 0]
		tmp[1] = (kepler_L(p, q) - L0)[0, 0]
		l += -dgdgTi @ tmp

		ret = y + dgT @ l
		return ret[0 : 2], ret[2 : 4]

	def explicit_euler_step(dt, p, q):
		newP, newQ = kepler_equation(p, q)

		newP = p + dt * newP
		newQ = q + dt * newQ
		return (newP, newQ)

	def explicit_euler_step_proj_H(dt, p, q, H0):
		newP, newQ = explicit_euler_step(dt, p, q)

		return project_to_H(newP, newQ, H0)
		
	def explicit_euler_step_proj_HL(dt, p, q, H0, L0):
		newP, newQ = explicit_euler_step(dt, p, q)

		return project_to_HL(newP, newQ, H0, L0)


	def symplectic_euler_step(dt, p, q):
		newP = p - dt * kepler_dHdq(p   , q)
		newQ = q + dt * kepler_dHdp(newP, q)
		return newP, newQ

	def symplectic_euler_step_proj_H(dt, p, q, H0):
		newP, newQ = symplectic_euler_step(dt, p, q)

		return project_to_H(newP, newQ, H0)

	def symplectic_euler_step_proj_HL(dt, p, q, H0, L0):
		newP, newQ = symplectic_euler_step(dt, p, q)

		return project_to_HL(newP, newQ, H0, L0)
		
	e = 0.6
	p0 = np.float64((0, np.sqrt((1 + e) / (1 - e))))
	q0 = np.float64((1 - e, 0))
	t0 = 0.0
	tn = 700.0
	n  = 700000
	dt = (tn - t0) / n
	L0 = kepler_L(p0, q0)[0, 0]
	H0 = kepler_H(p0, q0)[0, 0]
	
	output_every = 2
	print_every = 2000

	# exact solution(well, not)
	#u0 = np.zeros((4,), np.float64)
	#u0[0 : 2] = p0
	#u0[2 : 4] = q0
	#solver = solvers.RKE(solvers.classic_4(), kepler_adaptor, t0, u0)	
	#
	#i = 0
	#x = []
	#y = []
	#for i in range(n + 1):
	#	if i % output_every == 0:
	#		t, u = solver.get_state()
	#		x.append(u[2])
	#		y.append(u[3])
	#	if i % print_every == 0:
	#		print('t: ', solver.t)
	#	solver.evolve(t0 + i * dt, dt)
	#
	#fig = pyplot.figure()
	#ax = fig.add_subplot(1, 1, 1)
	#ax.plot(x, y)
	#pyplot.show()

	# explicit euler
	p = p0
	q = q0
	x = []
	y = []
	for i in range(n + 1):
		if i % output_every == 0:
			x.append(q[0])
			y.append(q[1])
		if i % print_every == 0:
			print('t: ', t0 + dt * i)
		p, q = symplectic_euler_step(dt, p, q)	

	fig = pyplot.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(x, y)
	pyplot.show()

def local_projection_test():
	# variable order for pendulum equation : p1, p2, q1, q2
	# variable order for pendulum equation in local coordinates: a, w

	def pend_equ(t, y):
		p1, p2, q1, q2 = y[0], y[1], y[2], y[3]

		pp = p1 * p1 + p2 * p2
		qq = q1 * q1 + q2 * q2
		l = (pp - q2) / qq

		y = np.zeros((4,), np.float64)
		y[0] = -q1 * l
		y[1] = -1 - q2 * l
		y[2] = p1
		y[3] = p2
		return y

	def pend_jac(t, y):
		p1, p2, q1, q2 = y[0], y[1], y[2], y[3]

		pp = p1 * p1 + p2 * p2
		qq = q1 * q1 + q2 * q2
		l = (pp - q2) / qq

		j = np.zeros((4, 4), np.float64)
		j[0, 0] = -2 * p1 * q1 / qq; j[0, 1] = -2 * p2 * q1 / qq; j[0, 2] = l * (q1 * q1 - q2 * q2) / qq; j[0, 3] = (q1 + 2 * l * q1 * q2) / qq;
		j[1, 0] = -2 * p1 * q2 / qq; j[1, 1] = -2 * p2 * q2 / qq; j[1, 2] = l * q1 * q2 / qq;             j[1, 3] = q2 / qq - l * (q1 * q1 - q2 * q2) / qq;
		j[2, 0] = 1.0; j[2, 1] = 0.0; j[2, 2] = 0.0; j[2, 3] = 0.0;
		j[3, 0] = 0.0; j[3, 1] = 1.0; j[3, 2] = 0.0; j[3, 3] = 0.0;
		return j


	def local_pend_equ(t, z):
		a, w = z[0], z[1]

		z = np.zeros((2,), np.float64)
		z[0] = w
		z[1] = -np.sin(a)
		return z

	def local_pend_jac(t, z):
		a, w = z[0], z[1]

		j = np.zeros((2, 2), np.float64)
		j[0, 0] = 0         ; j[0, 1] = 1;
		j[1, 0] = -np.cos(a); j[1, 1] = 0;
		return j

	def from_local(z):
		a, w = z[0], z[1]

		cosa = np.cos(a)
		sina = np.sin(a)

		y = np.zeros((4,), np.float64)
		y[0] = w * cosa 
		y[1] = w * sina
		y[2] = sina
		y[3] = -cosa
		return y


	def integrate(solver, t0, dt, n, output_every, print_every):
		ts = []
		us = []
		for i in range(n):
			t, u = solver.get_state()

			if i % output_every == 0:
				ts.append(t)
				us.append(u)

			if i % print_every == 0:
				print(t, ': ', u)

			solver.evolve(t0 + i * dt, dt)
		return ts, us


	# initial values & parameteres
	a0 = m.pi / 2
	w0 = 3
	
	z0 = np.float64((a0, w0))
	
	u0 = from_local(z0)

	t0 = 0.0
	tn = 50.0
	n = 2000

	output_every = 1
	print_every  = n // 50  

	dt = (tn - t0) / n

	# solvers
	pend_equ_classic_4 = solvers.RKE(solvers.classic_4(), pend_equ, t0, u0) # test only
	pend_equ_imp_2     = solvers.RKI_naive(solvers.implicit_midpoint_2(), pend_equ, pend_jac, solvers.NeutonSolver(1e-15, 50), t0, u0)

	pend_loc_equ_classic_4 = solvers.RKE(solvers.classic_4(), local_pend_equ, t0, z0)

	print('----Integrating pendulum equation with classic_4...----')
	ts, pend_classic_4     = integrate(pend_equ_classic_4, t0, dt, n, output_every, print_every)
	print()

	print('----Integrating pendulum equation with imp_2...----')
	ts, pend_imp_2         = integrate(pend_equ_imp_2, t0, dt, n, output_every, print_every)
	print()

	print('----Integrating local pendulum equation with classic_4...----')
	ts, pend_loc_classic_4 = integrate(pend_loc_equ_classic_4, t0, dt, n, output_every, print_every)
	pend_loc_classic_4 = list(map(from_local, pend_loc_classic_4))
	print()

	# plotting
	fig = pyplot.figure()

	# plot trajectories
	def q1q2(ys):
		q1 = []
		q2 = []
		for y in ys:
			q1.append(y[2])
			q2.append(y[3])
		return q1, q2

	ax = fig.add_subplot(1, 2, 1)
	ax.set_title('trajectory')
	q1, q2 = q1q2(pend_classic_4)
	ax.plot(q1, q2)
	q1, q2 = q1q2(pend_imp_2)
	ax.plot(q1, q2)
	q1, q2 = q1q2(pend_loc_classic_4)
	ax.plot(q1, q2)
	ax.legend(['c4', 'imp2', 'loc_c4'])

	# plot weak invariant conservation
	ax = fig.add_subplot(1, 2, 2)
	ax.set_title('weak invariant')
	invs = list(map(lambda x: x[2]**2 + x[3]**2, pend_classic_4))
	ax.plot(ts, invs)
	invs = list(map(lambda x: x[2]**2 + x[3]**2, pend_imp_2))
	ax.plot(ts, invs)
	invs = list(map(lambda x: x[2]**2 + x[3]**2, pend_loc_classic_4))
	ax.plot(ts, invs)
	ax.legend(['c4', 'imp2', 'loc_c4'])
	pyplot.show()
	

def main():
	# solve_first()
	# solve_second()
	# solve_third()
	# kepler_test()
	local_projection_test()

if __name__ == '__main__':
	main()

	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()