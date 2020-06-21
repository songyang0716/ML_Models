import numpy as np
import pandas as pd 
# Reference https://blog.csdn.net/baidu_39413110/article/details/105323757

class XGB:
	def __init__(self,
				 base_score=0.5,
				 max_depth=5,
				 n_estimators=5,
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
		self.trees = {}


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
		if bst_var is None:
			return None
		# print("The best feature curoff feature is {}, value is {}".format(bst_var, bst_cut))
		# return prediction value for each leave
		id_left = X.loc[X[bst_var]<bst_cut].index.tolist()
		w_left = - G_left_best / (H_left_best + self.reg_lambda)

		id_right = X.loc[X[bst_var]>=bst_cut].index.tolist()
		w_right = - G_right_best / (H_right_best + self.reg_lambda)

		w[id_left] = w_left
		w[id_right] = w_right

		tree_structure = {(bst_var,bst_cut):{}}
		tree_structure[(bst_var,bst_cut)][('left',w_left)] = self.xgb_tree(X.loc[id_left], w, m_depth+1)
		tree_structure[(bst_var,bst_cut)][('right',w_right)] = self.xgb_tree(X.loc[id_right], w, m_depth+1)
				
		return tree_structure


	def _grad(self, y_hat, y):
		'''
		first order derivative
		'''
		if self.objective == 'logistic':
			return np.exp(y_hat)/(1+np.exp(y_hat)) - y
		elif self.objective == 'linear':
			return y_hat - y
		else:
			raise KeyError('objective must be linear or logistic!')


	def _hessian(self, y_hat, y):
		'''
		second order derivative
		'''
		if self.objective == 'logistic':
			return np.exp(y_hat) / (1+np.exp(y_hat)) / (1+np.exp(y_hat))
		elif self.objective == 'linear':
			return np.array([1]*y.shape[0])
		else:
			raise KeyError('objective must be linear or logistic!')


	def fit(self, X, y):
		if X.shape[0]!= y.shape[0]:
			raise ValueError('X and Y must have the same length!')
		X = X.reset_index(drop=True)
		y = y.values
		y_hat = np.array([self.base_score] * y.shape[0])

		for i in range(self.n_estimators):
			# print('fitting tree {}...'.format(i+1))

			X['g'] = self._grad(y_hat, y)
			X['h'] = self._hessian(y_hat, y)

			f_t = pd.Series([0] * y.shape[0])
			self.trees[i+1] = self.xgb_tree(X, f_t, 1)
			y_hat = y_hat + self.learning_rate * f_t

			# print('tree {} fit done!'.format(i+1))
			
		# print(self.trees)



	def _get_tree_node_w(self, X, tree, w):
		'''
		recursive method, get the tree structure
		'''
		if tree:
			# self.trees[i+1] = self.xgb_tree(X, f_t, 1)
			# tree_structure[(bst_var,bst_cut)][('left',w_left)]
			k = list(tree.keys())[0]
			# first split point 
			var, cut = k[0], k[1]
			X_left = X.loc[X[var] <= cut]
			id_left = X_left.index.tolist()

			X_right = X.loc[X[var] > cut]
			id_right = X_right.index.tolist()

			for kk in tree[k].keys():
				if kk[0] == 'left':
					tree_left = tree[k][kk]
					w[id_left] = kk[1]
				elif kk[0] == 'right':
					tree_right = tree[k][kk]
					w[id_right] = kk[1]

			self._get_tree_node_w(X_left, tree_left, w)
			self._get_tree_node_w(X_right, tree_right, w)


	def predict_raw(self, X):        
		X = X.reset_index(drop='True')
		y_pred = pd.Series([self.base_score]*X.shape[0])

		for i in range(self.n_estimators):
			tree = self.trees[i+1]
			y_t = pd.Series([0] * X.shape[0])
			self._get_tree_node_w(X, tree, y_t)
			y_pred= y_pred + self.learning_rate * y_t
			
		return y_pred


def rmse(y_pred, y):
		return np.sqrt(np.mean((y-y_pred)**2))


def main():
	# print(xg_boost_estimator.max_depth)
	# print(xg_boost_estimator.features)

	# create some fake dataset to fit 
	sample_size = 1000
	x1 = np.random.uniform(-10, 10, sample_size)
	x2 = np.random.normal(-5, 5, sample_size)
	y = 3*x1**3 - 2*x2**2 + 10*x2 + np.random.normal(0, 1, sample_size)

	df = pd.DataFrame({'x1':x1,
					   'x2':x2,
					   'y':y})

	for i in [10, 20, 30, 50, 80, 100, 150, 200]:
		print(i)
		xg_boost_estimator = XGB(features=['x1','x2'], n_estimators=i)
		xg_boost_estimator.fit(df[['x1','x2']], df['y'])
		y_predict = xg_boost_estimator.predict_raw(df[['x1','x2']])
		print(rmse(y_predict, df['y'].values))



if __name__ == '__main__':
	main()




# {('x1', -5.8022087878512725): 

# 	{('left', -1798.3700302630143): None, 
# 	 ('right', 167.0952291469791):

# 		 {('x1', 6.356679718512126): 

# 		 	{('left', -119.16198598249656): 
# 		 		{('x1', 3.5960377655124702): 
# 		 			{('left', -233.98540941736556): 
# 		 				{('x2', -9.007117363813915): 
# 		 					{('left', -470.94526428813936): None,
# 		 					 ('right', -150.8009394292927): 
# 		 					 	   {('x1', -3.3151186981538476): 
# 		 					 		   {('left', -355.8265171170506): None, 
# 		 			                    ('right', -85.94393800614615): None}}}}, 
# 		 	        ('right', 220.9749285583095): None}}, 
# 	         ('right', 1769.3862630821384): None}
# 	     }
# 	}
# }

