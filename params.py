import os

dresden_csv = 'dresden.csv'
dresden_images_root = 'dresden'
images_db_path = 'dresden.npy'

patch_num = 25
patch_span = 256 * 5

patches_root = 'patches'
train_db_path = os.path.join(patches_root,'train.npy')
test_db_path = os.path.join(patches_root,'test.npy')
patches_db_path = os.path.join(patches_root,'patches.npy')

split_path = 'split.npy'

robust_root = 'robust'
robust_db_path = os.path.join(patches_root,'patches.npy')

compr_root = 'compr'
compr_img = os.path.join(compr_root,'dresden')
compr_patches = os.path.join(compr_root,'patches')

