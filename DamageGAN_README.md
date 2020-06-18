# DamageGAN - Complete documentation

This is the complete documentation to preprocess the xBD datasets (train/va/test) so that they have the expected format
required for training using the pix2pix model.

## Preprocess satellite images for the pix2pix GAN

The pix2pix model requires that the images are in a specific format in folders _A_, _B_ and _AB_
This process is different from the preparation for the classifier, since the Pix2Pixs needs fodler A, B, AB, with the
same filenames in each folder. Follow the steps below to download and preprocess the xBD dataset.

Steps:
1. Download the training set, test set and holdout set from https://xview2.org/download-links and put them in the
`datasets` folder.
2. Unpack each and delete the compressed folders.
3. Ideally put all train, val, and test in one folder under datasets, eg. xBD_datasets. We continue this doc by assuming
this structure:
```
./datasets/
    ...
    xBD_datasets/
        train/
        val/
        test/
```
4. From the root of the project: `cd xbd_data_preprocessing`
5. For each dataset, run `preprocess_GAN.sh` passing the following parameters:
    - Run `preprocess_GAN.sh` with the following parameters:
        - -p: the path of the dataset (relative path to the xBD_datasets folder)
        - -f: the name of the folder (usually 'train', 'val' or 'test')
        - -c: the output category - either 'train', 'val' or 'test'.
        - **Example**: `bash preprocess_GAN.sh -p ../datasets/xBD_datasets -f train -c train`

- `preprocess_GAN.sh` generates the folders `xBD_polygons_AB` and `xBD_polygons_AB_csv` in the datasets folder, which
contain the images and labels in the expected format for pix2pix. These are in the form of pairs of images {A,B},
where A and B are two different depictions of the same underlying scene. If you run it for the 3 datasets, you will have
this structure:

```
./datasets/
    xBD_datasets/
        xBD_polygons_AB/
            A/
            B/
            AB/
                train/
                    ...
                    4abf749e-db2e-44f3-86dc-dde2e4a23229.png
                    ...
                val/
                test/
        xBD_polygons_AB_csv/
            satellite_AB_labels.txt
```
Folders `A` and `B` have the same subfolders as `AB` and the same filenames, with the only difference that the images
in AB are paired.

For more information on how and why the data should be in this format, read `./docs/datasets.md`.

Now use the generated folder `AB` and the labels textfile `satellite_AB_labels.txt` as an input to the GAN.


## Evaluate GAN performance with the resnet18 classifier
After creating the data folders `xBD_polygons_AB` and `xBD_polygons_AB_csv`, we train resnet18 on all B images
(train, val) and test on the ones generated from the GAN.

The files for training and testing are in the `./classifiers` folder and we call them through the bash script
`./scripts/run_classifier.sh`. The files `./classifiers/train.py` and `./classifiers/test.py` are abstract enough
to work with any future classifier.

Call run_classifier.sh with the argument **train** for training & validation, which then stores the checkpoint for the best
validation model. Then call it with **test** to load the stored checkpoint and test on the newly generated images.

```bash
bash scripts/run_classifier.sh train
bash scripts/run_classifier.sh test

```

This should run without any other calls or modifications, given that the data folder structure is the same as mentioned
in the previous section. However, you can modify `scripts/run_classifier.sh` to change the parameters.

## Train GAN with concurrent classifier evaluation

After each epoch of GAN training, the trained classifier is called to predict the classes of the generated images.
The trained classifier weights are expected to be in `classifiers/checkpoints/resnet18_checkpoint.pth`.

To train the GAN, run:
```
bash scripts/pix2pix_satellite.sh
```

After each epoch, generated sample images from train and val will be saved under `interim_results/model_name`, and the
classification results will be saved under `gan_eval/gan_eval.csv`.

The concurrent classifier evaluation can be set off with the option `--classifier_evaluation False` in
`scripts/pix2pix_satellite.sh`.

To view plots of the accuracy, loss and F1 score written in the `gan_eval.csv` for the training and validation sets, run:
```
python plots/plot_gan_eval_metrics.py --csv_file ~/Downloads/gan_eval.csv
```
