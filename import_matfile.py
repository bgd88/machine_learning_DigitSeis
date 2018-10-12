import scipy.io
import glob

dataDir = "../../data/images/"
filenames = glob.glob(dataDir + "*.mat")

testFile = filenames[0]
f = scipy.io.loadmat(testFile)

# List all groups
print("Keys: %s" % f.keys())
