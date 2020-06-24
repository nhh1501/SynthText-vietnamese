import h5py
# filename = "results/SynthText (copy).h5"
filename = "data/dset.h5"

import h5py
db =  h5py.File(filename,"r")
segmap = db['seg']                    
depth = db['depth']
iamge = db['image']
imnames = sorted(db['image'].keys())  
# data = the_file["data"]
# att = data.keys()
print('a')