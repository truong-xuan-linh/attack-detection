import gdown
import os
def setup():
    if not os.path.isdir('models'):
        id = "10qHQw110MSJ-SpImIs0nPSuB_HH9-h1D"
        gdown.download_folder(id=id, output="models")
