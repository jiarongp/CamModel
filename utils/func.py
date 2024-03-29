import os
import urllib
import pandas as pd
import numpy as np
import fnmatch
import time
import matplotlib.pyplot as plt
import tikzplotlib
from tqdm import tqdm
from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
from skimage import io
from params import dresden_images_root, train_csv_path, patch_span, \
        patch_num, patches_root, patches_db_path

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

def split_info(train_list, val_list, test_list, model_list, total):
    info = []
    weights = []
    for model in model_list:
        train = fnmatch.filter(train_list, model + '*')
        print("{} in training set: {}.".format(model, len(train)))
        val = fnmatch.filter(val_list, model + '*')
        print("{} in validation set: {}.".format(model, len(val)))
        test = fnmatch.filter(test_list, model + '*')
        print("{} in test set: {}.\n".format(model, len(test)))
        info.append([train, val, test])
        #(1 / neg)*(total)/2.0 
        weights.append((1/len(train + val + test))*(total)/ 2.0)
    return info, weights

def split(img_list, model_list, patches_db_path):
    num_test = int(len(img_list) * 0.2)
    num_train = int(len(img_list) - num_test)
    num_val = int(num_train * 0.2)
    num_train = num_train - num_val

    shuffle_list = np.random.permutation(img_list)
    train_list = shuffle_list[0:num_train].tolist()
    val_list = shuffle_list[num_train:num_train + num_val].tolist()
    test_list = shuffle_list[num_train + num_val:].tolist()

    info, weights = split_info(train_list, val_list, test_list, model_list, total=len(img_list))

    patches_db = {
        'train': train_list,
        'val': val_list,
        'test':test_list,
    }

    np.save(patches_db_path, patches_db)

    return train_list, val_list, test_list, info, weights


def patchify(img_name, patch_span, pacth_size=(256, 256)):
    img = io.imread(img_name)
    if img is None or not isinstance(img, np.ndarray):
        print('Unable to read the image: {:}'.format(args['img_path']))

    center = np.divide(img.shape[:2], 2).astype(int)
    start = np.subtract(center, patch_span/2).astype(int)
    end = np.add(center, patch_span/2).astype(int)
    sub_img = img[start[0]:end[0], start[1]:end[1]]
    sub_img = np.asarray(sub_img)
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))
    return patches


def extract(args):
    # 'Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
    # Use PNG for losslessly storing images
    output_rel_paths = [os.path.join(args['data_set'], args['img_brand_model'],
                        os.path.splitext(os.path.split(args['img_path'])[-1])[0]+'_'+'{:02}'.format(patch_idx) + '.png')\
                        for patch_idx in range(args['patch_num'])]
    read_img = False
    for out_path in output_rel_paths:
        out_fullpath = os.path.join(args['patch_root'], out_path)
        # if there is no this path, then we have to read images
        if not os.path.exists(out_fullpath):
            read_img = True
            break
    if read_img:
        img_name = os.path.join(args['img_root'], args['img_path'])
        patches = patchify(img_name, args['patch_span']).reshape((-1, 256, 256))
        
        for out_path, patch in zip(output_rel_paths, patches):
            out_fullpath = os.path.join(args['patch_root'],out_path)
            # the diretory of the patches images
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir)
            if not os.path.exists(out_fullpath):
                io.imsave(out_fullpath, patch, check_contrast=False)
    return output_rel_paths

def patch(brand_model, path, data_set, patches_root=patches_root, patch_span=patch_span, \
          patch_num=patch_num, img_root=dresden_images_root):
    
    imgs_list = []

    for img_path in tqdm(path):
        imgs_list += [{'data_set':data_set,
                       'img_path':img_path,
                       'img_brand_model':brand_model,
                       'patch_span':patch_span,
                       'patch_num':patch_num,
                       'patch_root': patches_root,
                       'img_root': img_root
                       }]
    num_processes = 4
    
    pool = Pool(processes=num_processes)
    paths = pool.map(extract, imgs_list)

def evaluate(model_list, generator, model, index, columns, num_batch=100, title=None, tex=False):
    t0 = time.time()
    hist = [[0, 0, 0] for i in range(len(model_list))]
    conf = []
    real_labels = []
    pred_labels = []
    for i in range(num_batch):
        gen = next(generator)
        pred = model.predict(gen[0])
        pred_label = np.argmax(pred, axis=1)
        real_label = np.argmax(gen[1], axis=1)
        pred_labels.append(pred_label)
        real_labels.append(real_label)
        conf.append(pred)
        for j in range(len(pred_label)):
            hist[real_label[j]][pred_label[j]] += 1
    t1 = time.time()
    print('\nIt tooks {:d} seconds\n'.format(int(t1-t0)))

    df = pd.DataFrame(hist, index=index, columns=columns)
    print('columns are predictions, index are ground truth\n')
    display(df)
    ax = df.plot.bar(stacked=True, figsize=(10, 5), title=title)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if height==0: 
            continue
        else:
            x, y = p.get_xy() 
            ax.text(x+width/2, 
                    y+height/2, 
                    "{}".format(int(height)), 
                    horizontalalignment='center', 
                    verticalalignment='center')
    if tex == True:
        tikzplotlib.save("test_result.tex")
    return hist, conf, pred_labels, real_labels

def mean_error(conf, pred_labels, real_labels, real_model, pred_model):
    conf = np.vstack((conf))
    conf_max = np.amax(conf, axis=1)
    pred_labels = np.hstack((pred_labels))
    real_labels = np.hstack((real_labels))

    mean = np.zeros((len(real_model), len(pred_model)))
    var = np.zeros((len(real_model), len(pred_model)))
    for i in range(len(real_model)):
        for j in range(len(pred_model)): 
            c = conf_max[(pred_labels==j) & (real_labels==i)]
            if len(c) != 0:
                mean[i, j] = np.mean(c)
                var[i, j] = np.var(c)
            else:
                mean[i, j] = 0
                var[i, j] = 0

    print('The mean of the confidence is: \n')
    df_mean = pd.DataFrame(mean, index=real_model, columns=pred_model)
    display(df_mean)
    print('The standard deviation of the confidence is: \n')
    df_error = pd.DataFrame(var, index=real_model, columns=pred_model)
    display(df_error)
    
    return df_mean, df_error

def plot_conf(conf, pred_labels, real_labels, graph, real_model, pred_model):
    conf = np.vstack((conf))
    pred_labels = np.hstack((pred_labels))
    real_labels = np.hstack((real_labels))
    
    for i in range(len(real_model)):
        weights = []
        # find out the corresponding true labels
        idx = np.hstack(np.argwhere(real_labels==i))
        for j in range(len(pred_model)):
            # for each camera label, it has a list to store its weights
            weights.append([c[j] for c in conf[idx]])
        df = pd.DataFrame(np.array(weights).transpose(), columns=pred_model)
        df.plot(title='Classifying ' + real_model[i] + ' images', ax=graph[i])
#     plt.tight_layout()