import os
import fnmatch
import numpy as np
from skimage.util.shape import view_as_blocks
from skimage import io

def split_info(train_list, val_list, test_list, model_list):
    for model in model_list:
        tmp_list = fnmatch.filter(train_list, model + '*')
        print("{} in training set: {}.".format(model, len(tmp_list)))
        tmp_list = fnmatch.filter(val_list, model + '*')
        print("{} in validation set: {}.".format(model, len(tmp_list)))
        tmp_list = fnmatch.filter(test_list, model + '*')
        print("{} in test set: {}.\n".format(model, len(tmp_list)))

def split(img_list, model_list, patches_db_path):
    num_test = int(len(img_list) * 0.2)
    num_train = int(len(img_list) - num_test)
    num_val = int(num_train * 0.2)
    num_train = num_train - num_val

    shuffle_list = np.random.permutation(img_list)
    train_list = shuffle_list[0:num_train].tolist()
    val_list = shuffle_list[num_train:num_train + num_val].tolist()
    test_list = shuffle_list[num_train + num_val:].tolist()

    split_info(train_list, val_list, test_list, model_list)

    patches_db = {
        'train': train_list,
        'val': val_list,
        'test':test_list,
    }

    np.save(patches_db_path, patches_db)

    return train_list, val_list, test_list


def patchify(img_name, patch_span, pacth_size=(256, 256)):
    img = io.imread(img_name)
    if img is None or not isinstance(img, np.ndarray):
        print('Unable to read the image: {:}'.format(args['img_path']))

    center = np.divide(img.shape[:2], 2).astype(int)
    start = np.subtract(center, patch_span/2).astype(int)
    end = np.add(center, patch_span/2).astype(int)
    sub_img = img[start[0]:end[0], start[1]:end[1]]
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))
    return patches


def extract(args):
    # 'Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
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
                io.imsave(out_fullpath, patch)

    return output_rel_paths