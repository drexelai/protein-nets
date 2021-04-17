# Distance is measured in Angstroms

# Class represents a single grid point in a 3D window
class grid_point:
	def __init__(gp, occupancy = 0, atom = None, coords = None, aa = None, diangle = None, distance_to_nearest_atom = None, nearest_atom = None,threshold = 3, atoms_within_threshold = None, dm = None):

		# Logical value of whether an atom is present
		gp.occupancy = occupancy

		# Name of atom if present
		gp.atom = atom

		# Store atomic positions when atom is present and middle of grid when an atom is not
		gp.coords = coords

		# Name of residue if present
		gp.aa = aa

		# Dihedral angle if present
		gp.diangle = diangle

		# Distance to nearest atom if atom is present
		gp.distance_to_nearest_atom = distance_to_nearest_atom
		gp.nearest_atom = nearest_atom

		# Threshold and atoms within that threshold is atom is present
		gp.threshold = threshold
		gp.atoms_within_threshold = atoms_within_threshold