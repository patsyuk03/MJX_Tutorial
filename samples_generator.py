import os

# xla_flags = os.environ.get('XLA_FLAGS', '')
# xla_flags += ' --xla_gpu_triton_gemm_any=True'
# os.environ['XLA_FLAGS'] = xla_flags

import bernstein_coeff_order10_arbitinterval
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax


class SamplesGenerator():

	def __init__(self, num_dof=6, num_batch=100, num_steps=200, timestep=0.02):
		super(SamplesGenerator, self).__init__()
	 
		self.num_dof = num_dof
		self.num_batch = num_batch
		self.t = timestep
		self.num = num_steps

		self.t_fin = self.num*self.t
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.nvar_single = jnp.shape(self.P_jax)[1]
		self.nvar = self.nvar_single*self.num_dof 
  
		self.A_projection = jnp.identity(self.nvar)
		self.rho_ineq = 1.0
		self.rho_projection = 1.0
		
		A_v_ineq, A_v = self.get_A_v()
		self.A_v_ineq = jnp.asarray(A_v_ineq) 
		self.A_v = jnp.asarray(A_v)

		A_a_ineq, A_a = self.get_A_a()
		self.A_a_ineq = jnp.asarray(A_a_ineq) 
		self.A_a = jnp.asarray(A_a)
  
		A_p_ineq, A_p = self.get_A_p()
		self.A_p_ineq = jnp.asarray(A_p_ineq) 
		self.A_p = jnp.asarray(A_p)
  
		A_eq = self.get_A_eq()
		self.A_eq = jnp.asarray(A_eq)
  
		Q_inv = self.get_Q_inv(A_eq)
		self.Q_inv = jnp.asarray(Q_inv)
  
		A_theta, A_thetadot, A_thetaddot = self.get_A_traj()
		self.A_theta = jnp.asarray(A_theta)
		self.A_thetadot = jnp.asarray(A_thetadot)
		self.A_thetaddot = jnp.asarray(A_thetaddot)
		
		self.compute_boundary_vec_batch = (jax.vmap(self.compute_boundary_vec_single, in_axes = (0)  ))

		self.key= jax.random.PRNGKey(0)
		self.maxiter_projection = 10

		self.v_max = 0.8
		self.a_max = 1.8
		self.p_max = 180*np.pi/180
  
		self.l_1 = 1.0
		self.l_2 = 1.0
		self.l_3 = 1.0
  
	def get_A_traj(self):
		A_theta = np.kron(np.identity(self.num_dof), self.P )
		A_thetadot = np.kron(np.identity(self.num_dof), self.Pdot )
		A_thetaddot = np.kron(np.identity(self.num_dof), self.Pddot )
		return A_theta, A_thetadot, A_thetaddot	
	
	def get_A_p(self):
		A_p = np.vstack(( self.P, -self.P     ))
		A_p_ineq = np.kron(np.identity(self.num_dof), A_p )
		return A_p_ineq, A_p
	
	def get_A_v(self):
		A_v = np.vstack(( self.Pdot, -self.Pdot     ))
		A_v_ineq = np.kron(np.identity(self.num_dof), A_v )
		return A_v_ineq, A_v

	def get_A_a(self):
		A_a = np.vstack(( self.Pddot, -self.Pddot  ))
		A_a_ineq = np.kron(np.identity(self.num_dof), A_a )
		return A_a_ineq, A_a
	
	def get_A_eq(self):
		return np.kron(np.identity(self.num_dof), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], self.Pddot[-1]    )))
	
	def get_Q_inv(self, A_eq):
		Q_inv = np.linalg.inv(np.vstack((np.hstack(( np.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_v_ineq.T, self.A_v_ineq)+self.rho_ineq*jnp.dot(self.A_a_ineq.T, self.A_a_ineq)+self.rho_ineq*jnp.dot(self.A_p_ineq.T, self.A_p_ineq), A_eq.T)  ), 
									 np.hstack((A_eq, np.zeros((np.shape(A_eq)[0], np.shape(A_eq)[0])))))))	
		return Q_inv

	@partial(jax.jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_term):
		b_eq_term = state_term.reshape(5, self.num_dof).T
		b_eq_term = b_eq_term.reshape(self.num_dof*5)
		return b_eq_term

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection(self, lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples):
  
		v_max_temp = jnp.hstack(( self.v_max*jnp.ones((self.num_batch, self.num  )),  self.v_max*jnp.ones((self.num_batch, self.num  ))       ))
		v_max_vec = jnp.tile(v_max_temp, (1, self.num_dof)  )

		a_max_temp = jnp.hstack(( self.a_max*jnp.ones((self.num_batch, self.num  )),  self.a_max*jnp.ones((self.num_batch, self.num  ))       ))
		a_max_vec = jnp.tile(a_max_temp, (1, self.num_dof)  )
		
		p_max_temp = jnp.hstack(( self.p_max*jnp.ones((self.num_batch, self.num  )),  self.p_max*jnp.ones((self.num_batch, self.num  ))       ))
		p_max_vec = jnp.tile(p_max_temp, (1, self.num_dof)  )
  
		b_v = v_max_vec 
		b_a = a_max_vec 
		b_p = p_max_vec
		
		b_v_aug = b_v-s_v
		b_a_aug = b_a-s_a 
		b_p_aug = b_p-s_p
  
		lincost = -lamda_v-lamda_a-lamda_p-self.rho_projection*jnp.dot(self.A_projection.T, xi_samples.T).T-self.rho_ineq*jnp.dot(self.A_v_ineq.T, b_v_aug.T).T-self.rho_ineq*jnp.dot(self.A_a_ineq.T, b_a_aug.T).T-self.rho_ineq*jnp.dot(self.A_p_ineq.T, b_p_aug.T).T
		sol = jnp.dot(self.Q_inv, jnp.hstack(( -lincost, b_eq_term )).T).T
  
		primal_sol = sol[:, 0:self.nvar]
		s_v = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_v_ineq, primal_sol.T).T+b_v  )
		res_v = jnp.dot(self.A_v_ineq, primal_sol.T).T-b_v+s_v 

		s_a = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_a_ineq, primal_sol.T).T+b_v  )
		res_a = jnp.dot(self.A_a_ineq, primal_sol.T).T-b_a+s_a 

		s_p = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num*self.num_dof )), -jnp.dot(self.A_p_ineq, primal_sol.T).T+b_p  )
		res_p = jnp.dot(self.A_p_ineq, primal_sol.T).T-b_p+s_p 
	
		lamda_v = lamda_v-self.rho_ineq*jnp.dot(self.A_v_ineq.T, res_v.T).T
		lamda_a = lamda_a-self.rho_ineq*jnp.dot(self.A_a_ineq.T, res_a.T).T
		lamda_p = lamda_p-self.rho_ineq*jnp.dot(self.A_p_ineq.T, res_p.T).T
  
		res_v_vec = jnp.linalg.norm(res_v, axis = 1)
		res_a_vec = jnp.linalg.norm(res_a, axis = 1)
		res_p_vec = jnp.linalg.norm(res_p, axis = 1)
		
		res_projection = res_v_vec+res_a_vec+res_p_vec
		
		return primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection

	@partial(jax.jit, static_argnums=(0,))
	def compute_projection_filter(self, xi_samples, state_term):

		b_eq_term = self.compute_boundary_vec_batch(state_term)
		s_v = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_a = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		s_p = jnp.zeros((self.num_batch, 2*self.num_dof*self.num   ))
		lamda_v = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_a = jnp.zeros(( self.num_batch, self.nvar  ))
		lamda_p = jnp.zeros(( self.num_batch, self.nvar  ))
		
		for i in range(0, self.maxiter_projection):
			primal_sol, s_v, s_a, s_p,  lamda_v, lamda_a, lamda_p, res_projection  = self.compute_projection(lamda_v, lamda_a, lamda_p, s_v, s_a, s_p,b_eq_term,  xi_samples)
	 
		return primal_sol

	
	@partial(jax.jit, static_argnums=(0,))
	def compute_xi_samples(self, key, xi_mean, xi_cov ):
		key, subkey = jax.random.split(key)
		xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.003*jnp.identity(self.nvar), (self.num_batch, ))
		return xi_samples, key
	
	@partial(jax.jit, static_argnums=(0,))
	def generate_samples(self, 
					  init_pos=jnp.array([1.5, -1.8, 1.75, -1.25, -1.6, 0]), 
					  init_vel=jnp.zeros(6), 
					  init_acc=jnp.zeros(6),):
		
		theta_init = jnp.tile(init_pos, (self.num_batch, 1))
		thetadot_init = jnp.tile(init_vel, (self.num_batch, 1))
		thetaddot_init = jnp.tile(init_acc, (self.num_batch, 1))
		thetadot_fin = jnp.zeros((self.num_batch, self.num_dof))
		thetaddot_fin = jnp.zeros((self.num_batch, self.num_dof))

		state_term = jnp.hstack((theta_init, thetadot_init, thetaddot_init, thetadot_fin, thetaddot_fin))
		state_term = jnp.asarray(state_term)
		
		xi_mean = jnp.zeros(self.nvar)
		xi_cov = 10*jnp.identity(self.nvar)
  
		key, subkey = jax.random.split(self.key)

		xi_samples, key = self.compute_xi_samples(key, xi_mean, xi_cov)
		xi_filtered = self.compute_projection_filter(xi_samples, state_term)

		thetadot = jnp.dot(self.A_thetadot, xi_filtered.T).T

		return thetadot

def main():
	num_dof = 6
	num_batch = 500
	num_steps = 50
	timestep = 0.05
	sg = SamplesGenerator(num_dof=num_dof, num_batch=num_batch, num_steps=num_steps, timestep=timestep)
	trajectories = sg.generate_samples()

	np.savetxt(f"{os.path.dirname(__file__)}/samples/trajectories.csv",trajectories, delimiter=",")

if __name__=="__main__":
	main()
  	
