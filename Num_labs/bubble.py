import numpy as np
import math as m
import solvers as s

from matplotlib import pyplot


class BubbleModelBase:
	def __init__(self
		, surface_tension  # 

		, liquid_density   # 
		, liquid_viscosity # 
		, liquid_velocity  #
		, liquid_hpressure # initial pressure in the bubble

		, vapour_pressure

		, radius0, velocity0
		, k, gamma
		, frequency):
		self.sigma = surface_tension
		self.gamma = gamma

		self.pv = vapour_pressure

		self.p0   = liquid_hpressure
		self.eta  = liquid_viscosity
		#self.p    = liquid_pressure
		self.rho0 = liquid_density
		self.v    = liquid_velocity
	
		self.k  = k
		self.nu = frequency

		self.r0 = radius0
		self.v0 = velocity0


	def gas_pressure(self, t, u):
		p0, sig, pv, y = self.p0, self.sigma, self.pv, self.gamma
		r0, r = self.r0, u[0]

		return (p0 + 2 * sig / r0 - pv) * (r0 / r)**(3 * y)

	def inf_pressure(self, t):

		return self.p0 + self.acoustic_pressure(t)

	def acoustic_pressure(self, t):
		p0, k, nu = self.p0, self.k, self.nu

		return p0 * (1 + k * m.sin(2 * m.pi * nu * t))

	
	def deriv_acoustic_pressure(self, t):
		p0, k, nu = self.p0, self.k, self.nu

		return p0 * k * (2 * m.pi * nu) * m.cos(2 * m.pi * nu * t)

	def deriv_gas_pressure(self, t, u):
		p0, sig, pv, y = self.p0, self.sigma, self.pv, self.gamma
		r0, r, v = self.r0, u[0], u[1]

		return (p0 + 2 * sig / r0 - pv) * (r0 / r)**(3 * y - 1) * (-r0 / r**2 * v)


class RayleighPlessetModel(BubbleModelBase):
		def __init__(self
		, surface_tension  # 

		, liquid_density   # 
		, liquid_viscosity # 
		, liquid_velocity  #
		, liquid_hpressure # initial pressure in the bubble

		, vapour_pressure

		, radius0, velocity0
		, k, gamma
		, frequency):
			super(RayleighPlessetModel, self).__init__(
				surface_tension  # 

				, liquid_density   # 
				, liquid_viscosity # 
				, liquid_velocity  #
				, liquid_hpressure # initial pressure in the bubble

				, vapour_pressure

				, radius0, velocity0
				, k, gamma
				, frequency
			)

		def equ(self, t, u):
			res = np.zeros((2,), np.float64)

			pv, rho0, sigma, eta = self.pv, self.rho0, self.sigma, self.eta		
			r,v = u[0], u[1]
	
			pg = self.gas_pressure(t, u)
			pinf = self.inf_pressure(t)	

			res[0] = v
			res[1] = 1/r * (-3/2 * v**2 + 1/rho0 * (pg + pv - pinf - 2*sigma/r - 4*eta*v/r))

			return res

		def __call__(self, t, u):

			return self.equ(t, u)

class HerringFlynnModel(BubbleModelBase):
	def __init__(self
	, surface_tension  # 

	, liquid_density   #  
	, liquid_viscosity # 
	, liquid_velocity  #
	, liquid_hpressure # initial pressure in the bubble

	, vapour_pressure

	, radius0, velocity0
	, k, gamma
	, frequency
	, sound_velocity):
		super(HerringFlynnModel, self).__init__(
			surface_tension  # 

			, liquid_density   # 
			#, liquid_pressure  # 
			, liquid_viscosity # 
			, liquid_velocity  #
			, liquid_hpressure # initial pressure in the bubble

			, vapour_pressure

			, radius0, velocity0
			, k, gamma
			, frequency
		)
		self.c = sound_velocity

	def equ(self, t, u):
		res = np.zeros((2,), np.float64)

		p0, pv, rho0, sigma, eta, c = self.p0, self.pv, self.rho0, self.sigma, self.eta, self.c
		r, v = u[0], u[1]
	
		pg  = self.gas_pressure(t, u)
		dpg = self.deriv_gas_pressure(t, u)

		pinf = self.inf_pressure(t)	

		pa  = self.acoustic_pressure(t)
		dpa = self.deriv_acoustic_pressure(t)

		pr       = -2 * sigma / r - 4 * eta * v / r - pa + pg + pv
		dpr_part = +2 * sigma * v / r**2 + 4 * eta * (v / r)**2 - dpa + dpg 			

		res[0] = v
		res[1] = (-3/2 * v**2 + 2 * v**3 / c + (pr - p0) / rho0) / (r - 2 * r*v / c + 4 * eta/c/rho0) + dpr_part / (c*rho0 - 2 * v * rho0 + 4 * eta / r)

		return res

	def __call__(self, t, u):

		return self.equ(t, u)

class KellerMiksisModel(BubbleModelBase):
	def __init__(self
		, surface_tension  # 

		, liquid_density   #  
		, liquid_viscosity # 
		, liquid_velocity  #
		, liquid_hpressure # initial pressure in the bubble

		, vapour_pressure

		, radius0, velocity0
		, k, gamma
		, frequency
		, sound_velocity):
		super(KellerMiksisModel, self).__init__(
			surface_tension  # 

			, liquid_density   # 
			#, liquid_pressure  # 
			, liquid_viscosity # 
			, liquid_velocity  #
			, liquid_hpressure # initial pressure in the bubble

			, vapour_pressure

			, radius0, velocity0
			, k, gamma
			, frequency
		)
		self.c = sound_velocity

	def equ(self, t, u):
		res = np.zeros((2,), np.float64)

		p0, pv, rho0, sigma, eta, c = self.p0, self.pv, self.rho0, self.sigma, self.eta, self.c
		r, v = u[0], u[1]
	
		pg  = self.gas_pressure(t, u)
		dpg = self.deriv_gas_pressure(t, u)

		pinf = self.inf_pressure(t)	

		pa  = self.acoustic_pressure(t)
		dpa = self.deriv_acoustic_pressure(t)

		pr       = -2 * sigma / r - 4 * eta * v / r - pa + pg + pv
		dpr_part = +2 * sigma * v / r**2 + 4 * eta * (v / r)**2 - dpa + dpg 			

		res[0] = v
		res[1] = (-3/2 * v**2 + 1/2 * v**3 / c + (1 + v / c) * (pr - p0) / rho0) / (r - r*v / c + 4 * eta/c/rho0) + dpr_part / (c*rho0 - v *rho0 + 4 * eta / r)

		return res

	def __call__(self, t, u):

		return self.equ(t, u)

class GilmoreModel(BubbleModelBase):
	def __init__(self
		, surface_tension  # 

		, liquid_density   #  
		, liquid_viscosity # 
		, liquid_velocity  #
		, liquid_hpressure # initial pressure in the bubble

		, vapour_pressure

		, radius0, velocity0
		, k, gamma
		, frequency
		, sound_velocity):
		super(GilmoreModel, self).__init__(
			surface_tension  # 

			, liquid_density   # 
			#, liquid_pressure  # 
			, liquid_viscosity # 
			, liquid_velocity  #
			, liquid_hpressure # initial pressure in the bubble

			, vapour_pressure

			, radius0, velocity0
			, k, gamma
			, frequency
		)
		self.c = sound_velocity

	def equ(self, t, u):
		res = np.zeros((2,), np.float64)

		p0, pv, rho0, sigma, eta, c = self.p0, self.pv, self.rho0, self.sigma, self.eta, self.c
		r, v = u[0], u[1]
	
		pg  = self.gas_pressure(t, u)
		dpg = self.deriv_gas_pressure(t, u)

		pinf = self.inf_pressure(t)	

		pa  = self.acoustic_pressure(t)
		dpa = self.deriv_acoustic_pressure(t)

		pr       = -2 * sigma / r - 4 * eta * v / r - pa + pg + pv
		dpr_part = +2 * sigma * v / r**2 + 4 * eta * (v / r)**2 - dpa + dpg 			

		n = 7
		A = 3001 # TODO : 3001 bars for water only
		B = 3000 # TODO : 3000 bars for water only

		Ap1d7 = A**(1/7)
		nm1dn = (n - 1) / n
		ndnm1 = n / (n - 1)

		H = ndnm1 * Ap1d7 / rho0 * ((pr + B)**nm1dn- (p0 + B)**nm1dn)
		dH_part = 0.0

		C = m.sqrt(c**2 + (n - 1) * H)

		res[0] = v
		res[1] = 0

		return res

	def __call__(self, t, u):

		return self.equ(t, u)


def evaluate_model(solver, model, dt, n):
	r = []
	v = []
	p = []
	t = []
	for i in range(n):
		rv = solver.value()
		r.append(rv[0])
		v.append(rv[1])
		p.append(model.gas_pressure(i * dt, rv))
		t.append(i * dt)

		solver.evolve(i * dt, dt)

	return r, v, p, t
	

def evaluate_rayleigh_plesset_model(dt = 0.5e-9, N = 200000):
	model = RayleighPlessetModel(
		  0.000008 
		, 1000.0
		, 0.000021966 
		, 0.0
		, 1e5 ###
		, 0.0
		, 5e-5 ###
		, 0.0
		, 0.3
		, 1.4
		, 22000.0
	)

	rv = np.float64((5e-5, 0.0)) ###

	solver = s.RKE(s.classic_4(), model, 0.0, rv)

	return evaluate_model(solver, model, dt, N)

def evaluate_herring_flynn_model(dt = 0.5e-9, N = 200000):
	model = HerringFlynnModel(
		  0.00008
		, 1000.0
		, 0.000021966
		, 0.0
		, 1e5 ###
		, 0.0
		, 5e-5 ###
		, 0.0
		, 0.28
		, 1.4
		, 22000.0
		, 1500
	)

	rv = np.float64((5e-5, 0.0)) ###

	solver = s.RKE(s.classic_4(), model, 0.0, rv)

	return evaluate_model(solver, model, dt, N)

def evaluate_keller_miksis_model(dt = 0.5e-9, N = 200000):
	model = KellerMiksisModel(
		  0.0008
		, 1000.0
		, 0.000021966
		, 0.0
		, 1e5
		, 0.5e4
		, 5e-7
		, 0.0
		, 0.28
		, 1.4
		, 20000.0
		, 1500
	)

	rv = np.float64((5e-7, 0.0))

	solver = s.RKE(s.classic_4(), model, 0.0, rv)

	return evaluate_model(solver, model, dt, N)

def evaluate_gilmore_model(dt, N):

	pass


def main():
	r0, v0, p0, t0 = evaluate_rayleigh_plesset_model(0.25e-9, 1000000)
	r1, v1, p1, t1 = evaluate_herring_flynn_model(0.25e-9, 1000000)
	#r2, v2, p2, t2 = evaluate_keller_miksis_model(0.25e-9, 500000)


	fig = pyplot.figure()

	ax = fig.add_subplot(1, 2, 1)
	ax.plot(t0, r0, linewidth = 1, label = 'r-p', linestyle = '--')
	ax.plot(t1, r1, linewidth = 1, label = 'h-f', linestyle = ':')
	#ax.plot(t2, r2, linewidth = 1, label = 'k-m', linestyle = '--')
	ax.legend(loc = 1)

	ax = fig.add_subplot(1, 2, 2)
	ax.plot(t0, p0, linewidth = 1, label = 'r-p', linestyle = '--')
	ax.plot(t1, p1, linewidth = 1, label = 'h-f', linestyle = ':')
	#ax.plot(t2, p2, linewidth = 1, label = 'k-m', linestyle = '--')
	ax.legend(loc = 1)

	pyplot.show()

if __name__ == '__main__':
	main()