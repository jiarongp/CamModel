import os

dresden_csv = 'dresden.csv'
dresden_images_root = 'dresden'
train_csv_path = 'train.csv'
unseen_csv_path = 'unseen.csv'

patch_num = 25
patch_span = 256 * 5

patches_root = 'train'
patches_db_path = os.path.join(patches_root,'patches.npy')
weights_path = os.path.join(patches_root,'weights.csv')

unseen_root = 'unseen'

post_root = 'post'
compr_img = os.path.join(post_root,'compr')
noise_img = os.path.join(post_root,'noise')
blur_img = os.path.join(post_root,'blur')

ins_root = 'instance'
ins_patches_db = os.path.join(ins_root,'patches.npy')
ins_train_csv = os.path.join(ins_root, 'train.csv')
ins_test_csv = os.path.join(ins_root, 'test.csv')
ins_weights = os.path.join(ins_root,'weights.csv')
ins_train_path = os.path.join(ins_root,'train')
ins_test_path = os.path.join(ins_root,'test')

