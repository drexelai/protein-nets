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
x_min, y_min, z_min, x_max, y_max, z_max = 1, 10, 4, 64, 56, 63

atom_type = ['C', 'N', 'O', 'S', 'None']
atom_type_data = pd.Series(atom_type)
atom_type_encoder = np.array(pd.get_dummies(atom_type_data))

atom_pos = ['O1', 'C9', 'O3', 'CZ2', 'CG2', 'CG', 'NE1', 'C1', 'C2', 'N3', 'CZ', 'OE2', 'SE', 'OE1', 'ND1', 'NH2', 'CE', 'C', 'OE21', 'OD2', 'OG', 'CH2', 'OXT', 'C5', 'ND2', 'C13', 'OE12', 'SD', 'C4', 'O', 'C6', 'C7', 'CE3', 'CH1', 'CA', 'C11', 'CB', 'CE1', 'NZ', 'C3', 'C12', 'OE11', 'NE', 'NE2', 'OG1', 'OH', 'N2', 'OT1', 'N1', 'O2', 'C14', 'C8', 'CD1', 'CG1', 'OD1', 'N', 'C10', 'CD2', 'CZ3', 'NH1', 'S', 'OT2', 'OE22', 'CD', 'SG', 'CE2', 'O4', 'None']
atom_pos_data = pd.Series(atom_pos)
atom_pos_encoder = np.array(pd.get_dummies(atom_pos_data))
dataset_file = 'ptn11H_10'

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

def train_data_loader(files, feature_set, fdir='ptndata_10H/'):
	global atom_type, atom_type_encoder, atom_pos, atom_pos_encoder, x_min, y_min, z_min, x_max, y_max, z_max
	for q, file in tqdm(enumerate(files)):
		entry = pickle.load(open(fdir + file, 'rb'))
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
	fdir='ptn11H_10/'
	files = os.listdir(fdir)
	files.sort()

	#print(load_feature_dimensions(files, fdir))
	# Initialize the feature set
	feature_set = None
	if os.path.isfile(dataset_file+'.npy'):
		feature_set = np.load(dataset_file+'.npy')
	else:
		feature_set = np.zeros(shape=(len(files), z_max-z_min, y_max-y_min, x_max-x_min, 1 + len(atom_type) + len(atom_pos)))
		train_data_loader(files, feature_set, fdir=fdir)
		np.save(dataset_file, feature_set)
	# feature_set_ = np.array([[[[ [0] * (1 + len(atom_type) + len(atom_pos)) for i in range(x_min, x_max)] for j in range(y_min, y_max)] for k in range(z_min, z_max)] for q in range(validation_samples)])



