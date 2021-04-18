from loadptn import train_data_loader, z_max, z_min, y_max, y_min, x_max, x_min, atom_pos, atom_type
import os
import numpy as np
def get_data(fdir):
	files = os.listdir(fdir)

	#print(load_feature_dimensions(files, fdir))
	# Initialize the feature set
	feature_set = None
	if os.path.isfile(fdir+'.npy'):
		feature_set = np.load(fdir+'.npy')
	else:
		feature_set = np.zeros(shape=(len(files), z_max-z_min, y_max-y_min, x_max-x_min, 1 + len(atom_type) + len(atom_pos)))
		train_data_loader(files, feature_set, fdir=fdir)
		np.save(fdir, feature_set)

	return feature_set