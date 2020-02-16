import os
import urllib
import pandas as pd
import numpy as np
from skimage import io
from tqdm import tqdm

def download(data, images_root, csv_path):

    csv_rows = []
    path_list = []
    brand_model_list = []

    for i in range((data.shape[0])): 
        csv_rows.append(list(data.iloc[i, :]))

    count = 0
    for csv_row in tqdm(csv_rows):

        filename, brand, model = csv_row[0:3]
        url = csv_row[-1]

        file_path = os.path.join(images_root, filename)

        try:
            if not os.path.exists(file_path):
                print('Downloading {:}'.format(filename))
                urllib.request.urlretrieve(url, file_path)

            # Load the image and check its dimensions
            img = io.imread(file_path)

            if img is None or not isinstance(img, np.ndarray):
                print('Unable to read image: {:}'.format(filename))
                # removes (deletes) the file path
                os.unlink(file_path)

            # if the size of all images are not zero, then append to the list
            if all(img.shape[:2]):
                count += 1
                brand_model = '_'.join([brand, model])
                brand_model_list.append(brand_model) 
                path_list.append(filename)

            else:
                print('Zero-sized image: {:}'.format(filename))
                os.unlink(file_path)

        except IOError:
            print('Unable to decode: {:}'.format(filename))
            os.unlink(file_path)

        except Exception as e:
            print('Error while loading: {:}'.format(filename))
            if os.path.exists(file_path):
                os.unlink(file_path)

    print('Number of images: {:}'.format(len(path_list)))
    # create a images database as a dictionary
    df = pd.DataFrame({'brand_model': brand_model_list, 'path': path_list})
    df.to_csv(csv_path, index=False)
    print('Saving db to csv')