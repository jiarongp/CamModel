import os

dresden_csv = 'dresden.csv'
dresden_images_root = 'dresden'
train_csv_path = 'train.csv'
unseen_csv_path = 'unseen.csv'

patch_num = 25
patch_span = 256 * 5

patches_root = 'train'
train_db_path = os.path.join(patches_root,'train.npy')
test_db_path = os.path.join(patches_root,'test.npy')
patches_db_path = os.path.join(patches_root,'patches.npy')

split_path = 'split.npy'

unseen_root = 'unseen'

compr_root = 'compr'
compr_img = os.path.join(compr_root,'dresden')
compr_patches = os.path.join(compr_root,'patches')

ins_root = 'instance'
ins_csv = os.path.join(ins_root, 'instance.csv')
ins_patches = os.path.join(ins_root,'patches')
ins_patches_db = os.path.join(ins_patches,'patches.npy')
ins_test = os.path.join(ins_root,'test')

