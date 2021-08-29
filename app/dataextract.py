import bz2
import pickle
import _pickle as cPickle
import pandas as pd

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
 	cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    with bz2.open(file, "rb") as f:
    # Decompress data from file
        tempdata = f.read()
        with open("tempdata", "wb") as f:
    # Write compressed data to file
            f.write(tempdata)        
    data=pd.read_pickle("tempdata")
    return data






