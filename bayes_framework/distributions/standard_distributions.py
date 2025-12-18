from scipy.stats import uniform, loguniform, norm

class UniformDistributions:
    def __init__(self, a, b):
        self.left_bound  = a
        self.right_bound = b

    def pdf(self, x):
        return uniform.pdf(x, self.left_bound, self.right_bound)

    def log_pdf(self, x):
        return uniform.logpdf(x, self.left_bound, self.right_bound)

    def log_cdf(self, prob):
        return uniform.logcdf(prob, self.left_bound, self.right_bound)

    def cdf(self, prob):
        return uniform.cdf(prob, self.left_bound, self.right_bound)