import numpy as np
import scipy.sparse

class Settings:
	def __init__(self, N = 10, M = 50, K = 2, eta = 0.01, phi_density = 0.1, seed = 0):
		self.N = N
		self.M = M
		self.K = K
		self.eta = eta
		self.phi_density = phi_density
		self.seed = seed

def random_problem(settings = Settings()):
	np.random.seed(settings.seed)
	W = random_noise(settings.N, settings.K, settings.eta)
	H = np.random.rand(settings.N, settings.M)
	alpha = np.random.rand(settings.K, 1)
	phi = scipy.sparse.rand(settings.M, 1, density = settings.phi_density)
	theta = phi.dot(alpha.T)
	Y = H.dot(theta) + W
	return (H, phi, alpha, W, Y)

# generate random noise
# K columns of data, where each column is a random vector of dim N
def random_noise(N, K, eta):
	mean = np.zeros(N)
	cov = np.eye(N) * eta
	return np.random.multivariate_normal(mean, cov, K).T

print(random_problem())