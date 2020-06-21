import numpy as np
import pandas as pd 


class XGB:
	def __init__(self,
				 base_score=0.5,
				 max_depth=5,
				 n_estimators=10,
				 learning_rate=0.1,
				 reg_lambda=1,
				 gamma=0,
				 min_child_sample = None,
				 min_child_weight=1,
				 objective='linear'):
	
		self.base_score = base_score
		self.max_depth = max_depth
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.gamma = gamma
		self.min_child_sample = min_child_sample
		self.min_child_weight = min_child_weight
		self.objective = objective

	def xgb_tree(self, X, w, m_depth):
		"""
		X: training set
		w: weights for each sample
		m_depth: tree depth
		"""
		if m_depth > self.max_depth:
			return 
		bst_var, bst_cut = None, None 
		max_gain = 0
		G_left_best, G_right_best, H_left_best, H_right_best = 0,0,0,0

		# go through all the values of each feature, and record the maximum gain point



def main():
	xg_boost_estimator = XGB()
	print(xg_boost_estimator.max_depth)

	# create some fake dataset to fit 
	sample_size = 1000
	x1 = np.random.uniform(-10, 10, sample_size)
	x2 = np.random.normal(-5, 5, sample_size)
	y = 3*x1**3 - 2*x2**2 + 10*x2 + np.random.normal(0, 1, sample_size)

	df = pd.DataFrame({'x1':x1,
		 		       'x2':x2,
		 		       'y':y})
	print (df.shape)
	print (df.columns)




if __name__ == '__main__':
	main()