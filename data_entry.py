# Class used for storing individual protein structures between ptn.py and cnn.py
class data_entry:
	def __init__(de, mat=None, ordinal_features=None, one_hot_features=None, dm=None):
		de.mat = mat
		de.ordinal_features = ordinal_features
		de.one_hot_features = one_hot_features
		de.dm = dm
