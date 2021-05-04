import pandas as pd
import numpy as np
from numpy import asarray
from time import time

from data_entry import data_entry
from ptn_io import isfileandnotempty, getfileswithname
from grid_point import grid_point

import math
import random

import pickle
import os
from tqdm import tqdm

CUBIC_LENGTH_CONSTRAINT = 70
x_min, y_min, z_min, x_max, y_max, z_max = 17, 22, 13, 17+32, 22+32, 13+48

atom_type = ['C', 'N', 'O', 'S', 'None']
atom_type_data = pd.Series(atom_type)
atom_type_encoder = np.array(pd.get_dummies(atom_type_data))

atom_pos = ['CG', 'OG', 'OH', 'NH2', 'CE1', 'CD2', 'OG1', 'N', 'ND1', 'CD', 'SD', 'ND2', 'OD1', 'OE2', 'OE1', 'C', 'NE', 'OD2', 'SG', 'CB', 'CD1', 'CZ', 'NH1', 'CE2', 'CG1', 'NE2', 'NZ', 'CG2', 'CA', 'O', 'CE', 'None']
atom_pos_data = pd.Series(atom_pos)
atom_pos_encoder = np.array(pd.get_dummies(atom_pos_data))
dataset_file = 'ptn11H_10'
acceptable_protein_names = ['1aip18-28', '1ald78-88', '1bgy51-61', '1bkv10-21', '1c12109-119', '1cai146-156', '1d2y95-105', '1de488-98', '1deu198-208', '1djb255-265', '1dnu478-488', '1ds6142-152', '1dzt119-129', '1ej6870-880', '1eyv59-69', '1f9348-59', '1fbx62-72', '1fnu802-812', '1ga8125-135', '1gh0121-131', '1go4163-173', '1gwg10-20', '1h69116-126', '1hjv284-294', '1hk923-33', '1hl5116-126', '1hqj5-15', '1ilg199-209', '1irj50-60', '1irm99-109', '1iwo681-691', '1jqi534-544', '1k8i50-60', '1kei182-192', '1kfc75-85', '1lgb113-123', '1lk064-74', '1los132-142', '1lrh147-157', '1m5q42-52', '1mab51-61', '1mbu96-106', '1mlz125-135', '1mvw120-130', '1non126-136', '1o7t156-166', '1o7t73-83', '1og270-80', '1oi9170-180', '1oj7310-320', '1onx260-270', '1oow36-46', '1p1b86-96', '1q1g219-229', '1qe1253-263', '1qq1149-159', '1que77-87', '1qun19-29', '1rmt168-178', '1rqa16-26', '1rvx540-550', '1rzi126-136', '1syq203-213', '1tbu44-54', '1uxa296-306', '1v6o115-125', '1vge84-94', '1vio13-23', '1vqr87-102', '1vqu283-293', '1w5k24-34', '1wuh62-72', '1wxo22-32', '1xko123-133', '1y0w26-36', '1yde205-215', '1yi58-18', '1yk354-64', '1ynb33-43', '1z1j67-77', '1z6r45-56', '1zca187-197', '1zin196-206', '1zl21212-1222', '2a6969-79', '2a7w57-67', '2aaf351-361', '2aow168-178', '2ara115-125', '2bx4248-258', '2c10148-158', '2dxr408-418', '2e9b601-611', '2f1d125-135', '2fa472-82', '2fpt268-278', '2fsi280-290', '2g9t76-89', '2gpp393-403', '2gw1164-174', '2hnd250-260', '2hr3100-110', '2hyd110-120', '2i2q42-52', '2i2x327-337', '2if081-91', '2igk443-453', '2iou194-204', '2j1r43-53', '2j8840-50', '2j9615-25', '2jdh25-35', '2jj1257-267', '2nsp55-65', '2pff5-15', '2pw824-34', '2pzk268-278', '2q7l791-801', '2ql6106-116', '2qm1279-289', '2qr078-88', '2rhs205-215', '2rk750-60', '2uul62-72', '2uul75-85', '2v2l79-89', '2vef106-116', '2w7f258-268', '2wdr99-109', '2win588-598', '2wmy74-84', '2x5q98-108', '2xfe19-29', '2xin271-281', '2xla166-176', '2y5c68-78', '2yk320-30', '2yn4180-190', '2ynj55-65', '2yr5219-229', '2zaf281-291', '2zml73-83', '2zny59-69', '3aeq55-65', '3akf106-116', '3ao467-77', '3art542-552', '3azd11-21', '3bfg47-57', '3c2b158-168', '3c9121-31', '3cb9205-215', '3csm182-192', '3csu123-133', '3d2947-57', '3dqr404-414', '3ecy89-99', '3ee5303-313', '3euw285-295', '3evc141-151', '3exg344-354', '3f6s77-87', '3ftc50-60', '3fup865-875', '3fwu42-52', '3gw221-31', '3h1j20-30', '3hlr118-128', '3hws154-164', '3ias35-45', '3ic343-54', '3is817-27', '3ix039-49', '3jvy146-156', '3jwd169-179', '3k4l353-363', '3knb18-28', '3kpv184-194', '3kve321-331', '3kxf214-224', '3lin17-27', '3lke206-216', '3lkz79-89', '3lpl438-448', '3m6s54-64', '3m791-11', '3mbq90-100', '3muq56-66', '3n68368-378', '3ndo128-138', '3ngt111-121', '3oee139-149', '3om3234-244', '3owe27-37', '3pci425-435', '3q9n4-14', '3qu1121-131', '3qvz78-88', '3qxe76-86', '3r25292-302', '3r8b215-225', '3ril410-420', '3rq1328-338', '3ruc65-75', '3s1b31-41', '3s3823-33', '3so4286-296', '3t0y71-81', '3t2m103-113', '3t4a65-75', '3tky227-237', '3tr9205-215', '3tt2183-193', '3u2z324-334', '3u8j27-37', '3v5r106-116', '3vjh465-475', '3vng111-121', '3vop31-41', '3vrg9-19', '3w7v92-102', '3w8h386-396', '3wch120-130', '3wsa244-254', '3wsh80-90', '3wxf689-699', '3wyh96-106', '3x3c122-132', '3zlp106-116', '3zmf371-381', '3zou72-82', '3zpi336-346', '3zpj306-316', '3zvj111-121', '4b8c793-803', '4bls17-27', '4bt6101-111', '4c0c177-187', '4c0s228-238', '4c7r353-363', '4c90559-569', '4ckh92-102', '4cl7213-223', '4cs723-33', '4ctx382-392', '4d0m120-130', '4d1j215-225', '4dao105-115', '4das145-155', '4dr9148-158', '4dwz194-204', '4e52219-229', '4e5t109-119', '4eqc442-452', '4es5337-347', '4eux481-491', '4f0o2-12', '4f2p167-177', '4fd4100-110', '4fnp413-423', '4fq97-17', '4fxf504-514', '4gsl146-156', '4hac207-217', '4hb615-25', '4hg5192-202', '4hyr358-368', '4i1i130-140', '4igb346-356', '4inr63-73', '4itt21-31', '4jcr132-142', '4k1z83-93', '4kci558-568', '4ki776-86', '4knp178-188', '4knt88-98', '4ld7367-377', '4loc605-615', '4lrv49-62', '4lus85-95', '4lzw82-92', '4m0545-55', '4m0z397-407', '4m11408-418', '4mgk99-109', '4mpb378-388', '4mvm1148-1158', '4mz968-78', '4n4o177-187', '4n9i152-162', '4nrk143-153', '4nzv36-46', '4o6r293-303', '4ocl49-59', '4ojx105-115', '4oop49-59', '4ou8286-296', '4p9y84-94', '4po563-73', '4py3144-154', '4q7h191-201', '4qd8130-140', '4qux55-65', '4qz156-66', '4r41111-121', '4ree135-145', '4rhe95-105', '4rus63-73', '4to5379-389', '4uh4597-607', '4um3123-133', '4uud559-569', '4uwl837-847', '4v2n36-46', '4w7864-74', '4w8n187-197', '4w9h18-28', '4wjg171-181', '4wzb110-120', '4xcn67-77', '4xtw69-79', '4y8r28-38', '4ym1234-244', '4yqh692-702', '4z9o70-80', '4zci433-443', '4zgj180-190', '4zh4214-224', '4zlb137-147', '5a0q223-233', '5ad7459-469', '5afu303-313', '5aig108-118', '5apx36-46', '5apz98-108', '5aqv130-140', '5bps39-49', '5cax51-61', '5cpz41-51', '5d5o170-180', '5dy934-44', '5e7c26-36', '5eiz40-50', '5eno132-142', '5exf18-28', '5fj936-46', '5fli15-25', '5fm5587-597', '5fuc46-56', '5fue44-54', '5fw4171-181', '5h47101-111', '5hbb69-79', '5irg117-127', '5iw7770-780', '5j5g106-116', '5jh1176-186', '5jpi408-418', '5jsi62-72', '5kgt152-162', '5kyd17-27', '5l0f1029-1039', '5l0f1097-1107', '5l5q24-34', '5ldf321-331', '5lf446-56', '5mt310-20', '5nev178-188', '5nok636-646', '5nzr192-202', '5p8z91-101', '5poj131-141', '5pyc795-805', '5q37930-940', '5qm0170-180', '5szn395-405', '5tcg266-276', '5tja163-173', '5tou6-16', '5tv3148-158', '5u1552-62', '5udw69-79', '5ufl145-158', '5unr705-715', '5us6246-256', '5v9u92-102', '5vch133-143', '5vys96-106', '5w70312-322', '5w9a25-35', '5wbu1711-1721', '5wk1113-123', '5ws527-37', '5x8l89-99', '5xkg262-272', '5xnw165-175', '5xte165-175', '5xvi232-242', '5xxw304-314', '5y7292-102', '5yb441-51', '5yc8135-145', '5z0g84-94', '5z2l135-145', '5z2m67-77', '5zb4111-121', '5zb8172-182', '5zcp25-35', '5zwe285-295', '5zws5-15', '6a0n99-109', '6a2u75-85', '6aqh345-355', '6bed269-279', '6bgl25-35', '6brj840-850', '6bxb1-11', '6c2y532-542', '6cp095-105', '6dhi161-172', '6dnj194-204', '6dpv348-358', '6dpx50-60', '6e5b13-23', '6ea5153-163', '6eh1150-160', '6epc57-67', '6fdu255-265', '6flc365-375', '6fow415-425', '6g6b17-27', '6g6f9-19', '6gey93-103', '6gnq18-28', '6hb0134-144', '6hck14-24', '6huc61-71', '6hvw10-20', '6hw383-93', '6hwa78-88', '6hy08-18', '6i0x430-440', '6iwp67-77', '6jeb455-465', '6jeq65-75', '6jla177-187', '6jlm13-23', '6joh132-142', '6joh37-47', '6jwi13-23', '6jx5460-470', '6jy0433-443', '6ko717-27', '6l5618-28', '6l7c16-26', '6lrb509-519', '6m6665-75', '6m7g159-169', '6m8s50-60', '6mb275-85', '6mdv96-106', '6mfp7-17', '6mjg222-232', '6mo11058-1068', '6mtg261-271', '6mx2164-174', '6mx916-26', '6mx973-83', '6mya472-482', '6ner135-145', '6nl98-18', '6ofs74-84', '6oo2341-351', '6pbu189-199', '6pby275-285', '6pej178-188', '6pfa67-77', '6q0r311-321', '6qg3302-312', '6qsd15-25', '6r7l55-65', '6rco55-65', '6rdx103-113', '6re0191-201', '6re2113-123', '6rec113-123', '6rfc235-245', '6riq232-242', '6rya37-47', '6s79187-197', '6t23310-320', '6u65211-221', '6udu265-275', '6v16209-219', '6v2f134-144', '6vak166-176', '6vi4106-116', '6vnr628-638', '6vxc401-411', '6w9c64-74', '6w9d73-83']
# Given a set of files storing entry objects and their directory location, return their feature dimensions such as the positional atom types and the bounds for the matrix.
def load_feature_dimensions(files, fdir = 'ptndata_10H/'):
	x_min, y_min, z_min, x_max, y_max, z_max = CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, CUBIC_LENGTH_CONSTRAINT, 0, 0, 0
	atom_pos = []
	for i, file in enumerate(files):
		print('Percentage complete: ', round(i / len(files) * 100, 2), '%', sep='')
		entry = pickle.load(open(fdir + file, 'rb'))
		new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max = find_bounds(grid2logical(entry.mat))
		x_min, y_min, z_min, x_max, y_max, z_max = update_bounds(new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max, x_min, y_min, z_min, x_max, y_max, z_max)
		#print(f'x: [{x_min},{x_max}]\ty: [{y_min},{y_max}]\tx: [{z_min},{z_max}]')
		atom_pos = get_all_atoms(entry.mat, atom_pos)
	atom_pos.append('None')

	return atom_pos, x_min, y_min, z_min, x_max, y_max, z_max

def load_acceptable_dimensions(fdir = 'ptndata_10H/'):
	files = os.listdir(fdir)
	output = []
	x_min, y_min, z_min, x_max, y_max, z_max = 17, 22, 15, 49, 48, 52
	atom_pos =  ['CG', 'OG', 'OH', 'NH2', 'CE1', 'CD2', 'OG1', 'N', 'ND1', 'CD', 'SD', 'ND2', 'OD1', 'OE2', 'OE1', 'C', 'NE', 'OD2', 'SG', 'CB', 'CD1', 'CZ', 'NH1', 'CE2', 'CG1', 'NE2', 'NZ', 'CG2', 'CA', 'O', 'CE', 'None']
	for f in tqdm(files):
		entry = pickle.load(open(fdir + f, 'rb'))
		new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max = find_bounds(grid2logical(entry.mat))
		if x_min <= new_x_min and y_min <= new_y_min and z_min <= new_z_min and new_x_max <= x_max and new_y_max <= y_max and new_z_max <= z_max:
			new_atom_pos = get_all_atoms(entry.mat, [])
			no_new_protein_pos = True
			for pos in new_atom_pos:
				if pos not in atom_pos:
					no_new_protein_pos = False
			if no_new_protein_pos:
				output.append(f)
				print(f)
	return output

# This is almost like sample_gen, except it is a function instead of a generator function. This is used for generating the validation data before training the CNN. It generates the validation samples for all three of the metrics.
def sample_loader(files, feature_set_, atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, energy_scores, x_min, y_min, z_min, x_max, y_max, z_max, fdir='ptndata_10H/'):
#if True:
	y_rosetta = []
	y_mse = []
	y_dm = []
	for q, file in enumerate(files):
		print('Percentage complete: ', round(q / len(files) * 100, 2), '%', sep='')
		entry = pickle.load(open(fdir + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat, atom_type, atom_type_encoder)
		c = grid2atom(entry.mat, atom_pos, atom_pos_encoder)
#
		#y = np.reshape(y, (len(y), len(y[0][0]), len(y[0][0])))
		#y = y.astype(float)
		y_rosetta.append(energy_scores.loc['ptndata_10H/' + file]['rosetta_score'])
		y_mse.append(energy_scores.loc['ptndata_10H/' + file]['mse_score'])
		y_dm.append(entry.dm)
		for i in range(len(feature_set_[0])):
			for j in range(len(feature_set_[0][0])):
				for k in range(len(feature_set_[0][0][0])):
					feature_set_[q][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()

	y_rosetta = np.array(y_rosetta)
	y_rosetta = y_rosetta.reshape(-1,1)		

	y_mse = np.array(y_mse)
	y_mse = y_mse.reshape(-1,1)	

	y_dm = np.reshape(y_dm, (len(y_dm), len(y_dm[0][0]), len(y_dm[0][0])))
	y_dm = y_dm.astype(float)

	return feature_set_, y_rosetta, y_mse, y_dm


def select_region_dm(dm, shape):
	return np.array([[ [dm[k][j][i] for i in range(shape[1])] for j in range(shape[0])] for k in range(len(dm))])

# Given an object loaded matrix of grid points, return a logical matrix representing atomic positions
def grid2logical(mat):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				mat_[i][j][k] = mat[i][j][k].occupancy
	return mat_


# Given an object loaded matrix of grid points, return a matrix of atom types into general categories {'N', 'O', 'C', 'S'}
def grid2atomtype(mat, atom_type, atom_type_encoder):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is None:
					mat_[i][j][k] = atom_type_encoder[atom_type.index("None")]
				else:
					mat_[i][j][k] = atom_type_encoder[atom_type.index(atom[:1])]
	return mat_


# Given an object loaded matrix of grid points, return a matrix of specific atom types
def grid2atom(mat, atom_pos, atom_pos_encoder):
	a = len(mat)
	mat_ = [[[ [] for _ in range(a)] for _ in range(a)] for _ in range(a)]

	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is None:
					mat_[i][j][k] = atom_pos_encoder[atom_pos.index("None")]
				else:
					mat_[i][j][k] = atom_pos_encoder[atom_pos.index(atom)]
	return mat_


# Given an object loaded matrix of grid points, return a list of unique atoms.
def get_all_atoms(mat, atoms):
	for i in range(len(mat)):
		for j in range(len(mat[0])):
			for k in range(len(mat[0][0])):
				atom = mat[i][j][k].atom
				if atom is not None:
					atoms.append(atom)
	return list(set(atoms))


# Given a matrix, return the minimum required dimensions in order to capture all non-zero values.
def find_bounds(mat):
	x = [i for i in range(CUBIC_LENGTH_CONSTRAINT) if (np.array(mat[i]) != 0.0).any()]
	x_min = min(x)
	x_max = max(x)

	y = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) if (np.array(mat[j][i]) != 0.0).any()]
	y_min = min(y)
	y_max = max(y)

	z = [i for i in range(CUBIC_LENGTH_CONSTRAINT) for j in range(x_min, x_max) for k in range(y_min, y_max) if (np.array(mat[j][k][i]) != 0.0).any()]
	z_min = min(z)
	z_max = max(z)

	return x_min, y_min, z_min, x_max, y_max, z_max


# Given new bounds and old bounds, return the proper updated bounds.
def update_bounds(new_x_min, new_y_min, new_z_min, new_x_max, new_y_max, new_z_max, x_min, y_min, z_min, x_max, y_max, z_max):
	if new_x_min < x_min:
		x_min = new_x_min

	if new_y_min < y_min:
		y_min = new_y_min

	if new_z_min < z_min:
		z_min = new_z_min

	if new_x_max > x_max:
		x_max = new_x_max

	if new_y_max > y_max:
		y_max = new_y_max

	if new_z_max > z_max:
		z_max = new_z_max	

	return x_min, y_min, z_min, x_max, y_max, z_max

def train_data_loader(files, feature_set, fdir='ptndata_10H'):
	global atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, x_min, y_min, z_min, x_max, y_max, z_max
	for q, file in tqdm(enumerate(files)):
		entry = pickle.load(open(fdir +'/' + file, 'rb'))
		a = grid2logical(entry.mat)
		b = grid2atomtype(entry.mat, atom_type, atom_type_encoder)
		c = grid2atom(entry.mat, atom_pos, atom_pos_encoder)
		#y = energy_scores.loc['ptndata_10H/' + file]['mse_score']
		#y = np.array(y)
		#y = y.reshape(-1,1)	
		for i in range(len(feature_set[0])):
			for j in range(len(feature_set[0][0])):
				for k in range(len(feature_set[0][0][0])):
					feature_set[q][i][j][k] = [a[x_min + i][y_min + j][z_min + k]] + b[x_min + i][y_min + j][z_min + k].tolist() + c[x_min + i][y_min + j][z_min + k].tolist()

if __name__ == "__main__":
	fdir='ptn11H_1000'
	files = acceptable_protein_names
	# files = os.listdir(fdir)
	# files.sort()

	# print(load_feature_dimensions(files, fdir))
	# # Initialize the feature set
	# feature_set = None
	if os.path.isfile(fdir+'.npy'):
		feature_set = np.load(fdir+'.npy')
	else:
		feature_set = np.zeros(shape=(len(files), z_max-z_min, y_max-y_min, x_max-x_min, 1 + len(atom_type) + len(atom_pos)))
		train_data_loader(files, feature_set, fdir=fdir)
		np.save(fdir, feature_set)
	# feature_set_ = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(validation_samples)])



