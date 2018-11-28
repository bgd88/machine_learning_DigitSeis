import glob
from myPath import dataDir
import h5py
import pandas as pd

def return_images_withPixelCrossings(N):
    pickleFile = "./out/crossCounts.pkl"
    try:
        df = pd.read_pickle(pickleFile)
        print("Loaded Values from {}".format(pickleFile))
    except:
        print("Counting Pixel Crossings...")
        filenames = glob.glob(dataDir + "images/*.mat")
        use = []
        for ii in np.arange(0, len(filenames)):
            if ii%100==0:
                print(ii)
            filename = filenames[ii]
            try:
                f = h5py.File(filename, 'r')
                pixID = f['pixID'].value.astype('int')[0]
                unique, counts = np.unique(pixID, return_counts=True)
                if len(counts)==4:
                    use.append([filename, counts[-1]])
            except:
                print("Failed for File: {}".format(filename))
                continue
            f.close()

        # crossCounts = np.array(use)
        df = pd.DataFrame(use)
        df.columns = ['filename', 'counts']
        df = df.sort_values('counts', ascending=False)
        df.to_pickle(pickleFile)
        print("Wrote to {}".format(pickleFile))

    return df[0:N].filename.values.tolist()
