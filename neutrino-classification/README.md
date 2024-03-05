# Neutrino flavor classification using CNNs

## Rice multiclass training from pre-saved weights 

- In the dfs (testset and trainset) you will need to change the image_path prefix on all of the lines to point to where you are actually storing your pixel maps. I believe you will do the following:
```
df['image_path'] = df['image_path'].str.replace('/home/sophiaf/pixel_maps_val/', '/new/directory/')
```
- In Rice_ResNet_multiclass_fromsaved.py: sure you CTRL+F: "CHANGE PATH". That will take you to any hard-coded directories you will need to change, including where the dfs are stored, the model weights I've already trained, and any places you'll save results. 
- Then you can run the following: 
```
python Rice_ResNet_multiclass_fromsaved.py --num_epochs 20 --learning_rate 3e-4 --batch_size 64 --pixel_map_size 200 --listname 'urllist_0_1_2_10_11_12' --test_name 'ResNet_v2_20240110_urllist0_1_2_10_11_12' --path_checkpoint /path/to/m20240109
```
