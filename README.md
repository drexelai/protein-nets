# protein-nets

## Project objective

- Explore protein encoding (https://github.com/ethanmoyer/ptnstrerrpredict/blob/master/ptn.py)
- Data set is through Protein Data Bank through API calls (https://www.wwpdb.org/)

## Paper 1: Functional Protein Structure Annotation using a Deep Colvolution Generative Adversarial Network
- Create a deep convolution generalized advesarial network (DCGAN) to generate and discriminate between functional and non-functional protein structures.

Conference: https://www.bhi-bsn-2021.org/

## Paper 2: Novel Protein Structure Generation according to Specific Function using a Deep Colvolution Generative Adversarial Network
- Use a DCGAN to generate and discriminate between proteins with a particular function, such as ligand binding, RNA cleaving, etc

## Paper 3: Optiminal Protein Structure Folding using a Noisy Autoencoder
- Iteratively train and test an autoencoder network on increadinly messed-up decoy protein structures and explore the extent to which an autoencoder can refold them.

## Tasks
[x] - Define protein structures of interest, search space, and data base focus () \
[x] - Build database using list of four-letter protein structure IDs () \
[] - Download proteins from PDB and encode protein structures into grid structure (Ethan, ) \
[x] - Determine feature set based on encoding and decide if any more features should be included () \
[] - Train GAN/DCGAN on encoded protein structures () \
[] - Perform experiemnts/gather results to determine how well the model performed () \
[] - Explore the explainability of the model () \
[] - Generate figures for paper and sample test cases to illistrate how the model performs on an individual protein structure ()

## Paper #1 roles
Ethan: Data set, protein encoding, 
Isamu: Results
Jeff: background/GAN architecture, torsion angle
Mali: Introduction
Alisha: Background of protein structure prediction, abstract
Yigit: Related work/GAN architecture 
Adam: Background of protein structure prediction

## Agenda
 
### Plan for April 5th, 2021
- Introduce GANs (Jeff)
- Introduce protein structure encoding, prediction, mapping (Ethan)
  - Measuring Protein Structure using Machine Learning:
  - https://www.mabc2020.com/post/measuring-protein-structure-deviation-using-machine-learning

### Plan for April 14th, 2021
- Introduce GitHub for project management (Yigit)
- Assign taks for paper based on individual interest (Ethan)
  - Paper:
  - https://www.overleaf.com/read/nkfdszdmkjqf

### Plan for April 21st, 2021
- Aim to have paper #1 on DCGAN for functional vs nonfunctional protein structure annotation done and submitted by Sunday (18 April 2021)
- Discuss autoencoder project and more functional annotation of protein

## Resources
- https://papers.nips.cc/paper/2018/file/afa299a4d1d8c52e75dd8a24c3ce534f-Paper.pdf
- https://arxiv.org/pdf/2004.07119.pdf
- https://arxiv.org/pdf/1511.06434.pdf
- https://proteinstructures.com/structure/ramachandran-plot/
