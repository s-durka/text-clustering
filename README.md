**Problem description**
The purpose of the program is to cluster images of various characters.

**Dataset specifics**
link to dataset:
https://www.dropbox.com/scl/fo/wp4iz69odzi8ldnplwp0h/h?rlkey=udboc94aueqzs5rom9fpcjfvm&dl=0

-- The images contain images of various resolutions representing characters (or character sequences) of various sizes, thickness, style etc.
-- Unknown number of decision classes - in addition to regular letters, there are fragments of characters and characters glued together. 
-- Some letters may remain indistinguishable, eg "I", "|" and "1" or "m" and "rn".

**Solution**
The solution uses the K-Means algorithm to cluster the input images. 
Firstly, it normalizes the data by changing images to numpy arrays of greyscale values 0-255 of set dimensions, then it flattens the 2D array into a 1D vector.
Second, it applies PCA to reduce the dimensionality of the problem.
Lastly, it finds the optimal number of clusters 'n' by comparing silhouette scores for re-runs of the algorithm, and produces clustering with the best chosen number of 'n'.

If the number of clusters is too low, images representing the same letter may be labeled as different (split into different clusters). On the other hand, if there are too few clusters, images of two different letters will end up being classified as the same kind. The presented solution tries to balance the two issues.

The distance metric used is the standard Euclidean distance, as the K-Means algorithm is used.

-------------------------------------------------------------
                How to run:
-------------------------------------------------------------

python3 -m venv py_env
source ./py_env/bin/activate
pip install -r requirements.txt
python3 ./cluster.py <path to file with listed paths to images*>

*as shown in example.txt
