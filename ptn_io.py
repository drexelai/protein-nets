import os
from os import listdir
from os.path import isfile, join

def isfileandnotempty(file):
	return os.path.isfile(file) and os.stat(file).st_size != 0

def getfileswithname(fdir, names):			# make name a list
	return [f for f in listdir(fdir) if isfile(join(fdir, f)) and all([x in f for x in names])]