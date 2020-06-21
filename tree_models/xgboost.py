import numpy as np
import pandas as pd 
# Reference https://blog.csdn.net/baidu_39413110/article/details/105323757

class XGB:
	def __init__(self,
				 base_score=0.5,
				 max_depth=5,
				 n_estimators=10,
				 learning_rate=0.1,
				 reg_lambda=1,
				 gamma=0,
				 min_child_sample = 10,
				 min_child_weight=1,
				 features=None,
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
		self.features=features

	def xgb_tree(self, X, w, m_depth):
		"""
		X: training set
		w: weights for each sample / leave values for each sample
		m_depth: tree depth
		"""
		if m_depth > self.max_depth:
			return None
		bst_var, bst_cut = None, None 
		max_gain = 0
		n_sample = X.shape[0]
		G_left_best, G_right_best, H_left_best, H_right_best = 0,0,0,0
		# go through all the values of each feature, and record the maximum gain point
		for feature in self.features:
			for cut_value in np.unique(X[feature].values):
				if self.min_child_sample:
					left_bin_size = np.sum(X[feature] <= cut_value)
					if (left_bin_size < self.min_child_sample) or (n_sample - left_bin_size < self.min_child_sample):
						continue
	
				G_left  = np.sum(X.loc[X[feature] <= cut_value,'g'])
				G_right = np.sum(X.loc[X[feature] > cut_value,'g'])
				
				H_left  = np.sum(X.loc[X[feature] <= cut_value,'h'])
				H_right = np.sum(X.loc[X[feature] > cut_value,'h'])

				#https://stats.stackexchange.com/questions/317073/explanation-of-min-child-weight-in-xgboost-algorithm
				if self.min_child_weight:
					if (H_left < self.min_child_weight) or (H_right < self.min_child_weight):
						continue
				cur_gain = 0.5*(G_left**2/(H_left+self.reg_lambda) +
								G_right**2/(H_right+self.reg_lambda) -
							   (G_left+G_right)**2/(H_left+H_right+self.reg_lambda)) - self.gamma
				if cur_gain > max_gain:
					max_gain = cur_gain
					bst_var = feature
					bst_cut = cut_value
					G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right

		# if we failed to find the best split point
		if best_var is None:
			return None
		print("The best feature curoff feature is {}, value is {}".format(bst_var, bst_cut))
		# return prediction value for each leave
		id_left = X.loc[X[best_var]<best_cut].index.tolist()
		w_left = - G_left_best / (H_left_best + self.reg_lambda)

		id_right = X.loc[X[best_var]>=best_cut].index.tolist()
		w_right = - G_right_best / (H_right_best + self.reg_lambda)

		w[id_left] = w_left
		w[id_right] = w_rights

		tree_structure = {(best_var,best_cut):{}}
		tree_structure[(best_var,best_cut)][('left',w_left)] = self.xgb_tree(X.loc[id_left], w, m_dpth+1)
		tree_structure[(best_var,best_cut)][('right',w_right)] = self.xgb_tree(X.loc[id_right], w, m_dpth+1)
				
		return tree_structure


	def _grad(self, y_hat, Y):
		'''
		first order derivative
		'''
		if self.objective == 'logistic':
			return np.exp(y_hat)/(1+np.exp(y_hat)) - Y
		elif self.objective == 'linear':
			return y_hat - Y
		else:
			raise KeyError('objective must be linear or logistic!')


	def _hessian(self, y_hat, Y):
		'''
		second order derivative
		'''
		if self.objective == 'logistic':
			return np.exp(y_hat) / (1+np.exp(y_hat)) / (1+np.exp(y_hat))
		elif self.objective == 'linear':
			return np.array([1]*Y.shape[0])
		else:
			raise KeyError('objective must be linear or logistic!')

	def fit(self, X, Y):
		return 




def main():
	xg_boost_estimator = XGB(features=['x1','x2'])
	# print(xg_boost_estimator.max_depth)
	# print(xg_boost_estimator.features)

	# create some fake dataset to fit 
	sample_size = 100
	x1 = np.random.uniform(-10, 10, sample_size)
	x2 = np.random.normal(-5, 5, sample_size)
	y = 3*x1**3 - 2*x2**2 + 10*x2 + np.random.normal(0, 1, sample_size)

	df = pd.DataFrame({'x1':x1,
					   'x2':x2,
					   'y':y})
	# print (df.shape)
	# print (df.columns)
	# feature = 'x1'
	# print (np.unique(df[feature].values))

	xg_boost_estimator.xgb_tree(df,
								w=None,
								m_depth=0)




if __name__ == '__main__':
	main()