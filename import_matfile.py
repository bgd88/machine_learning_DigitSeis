#!/usr/bin/env python3
import scipy.io
import glob
from myPath import dataDir


filenames = glob.glob(dataDir + "*.mat")

testFile = filenames[0]
f = scipy.io.loadmat(testFile)

# List all groups
print("Keys: %s" % f.keys())

# Lets ge the S Structure
S = f['S']

# Print fields
print(S.dtype)
