# ANNA-PALM

Accelerating Single Molecule Localization Microscopy with Deep Learning.

For more details, please checkout https://annapalm.pasteur.fr/

## INSTALLATION

ANNA-PALM has the following dependencies:
 * Python 3.6
 * Pillow
 * numpy
 * scipy
 * matplotlib
 * scikit-image
 * tensorflow==1.1.0
 * imreg_dft
 * pytest

The recommended way to get python3.6 is to use ANACONDA, go the website: https://www.continuum.io/downloads ,
and download Anaconda(Python 3.6), then install it.

The next step is to install the dependencies, mainly for tensorflow.
Open a terminal, switch to the ANNA-PALM source code folder.
```bash
git clone https://github.com/imodpasteur/ANNA-PALM
cd ANNA-PALM
```

For a quick test running on CPU or you don't have a cuda compatible GPU, run the following command:
```bash
pip install -r requirements.txt
```

Or if you have an tensorflow compatible GPU, and you want to use GPU for training, run the following command:
```bash
pip install -r requirements-gpu.txt
```

To test your installation, you could run the following code:
```
cd ANNA-PALM
python run.py --workdir=./tmp_test
```
Once it starts without any error, you can terminate it with CTRL-C, otherwise, it will take a long time to actually finish the training.

## ImageJ plugin
Besides the python code, an ImageJ plugin for applying trained model can be downloaded [here](https://s3.eu-west-2.amazonaws.com/anna-palm-model/ANNA_PALM_Process-latest.jar).

## USAGE

## Train with simulated images

Two type of scripts are available, the first one is to run on simulated microtubules and nuclear pores.

For example, you can train a new ANNA-PALM model with the following command:
```
python run.py --workdir=./results/simulated_model
```

You can run the following command to see all the arguments.
```
# for example
python run.py --help
```

## Train with localization tables
 * In order to train an ANNA-PALM model with your own data, you need to prepare your localization tables. Currently, you need to use the csv format produced with [ThunderSTORM](https://github.com/zitmen/thunderstorm). If you don't have any data for now, you can [download](https://www.dropbox.com/sh/lwl1l3tdtzdr1re/AACmm8hRYszNVXwI0gqIeaoLa?dl=0) our microtubule data we used in the paper.
 * Create a folder as your working directory(for example `training_workdir_exp1`) , inside your `training_workdir_exp1`, you need to create a folder named `train`.
 * Then, place all your csv files into the `train` folder. Optionally, you could reserve one or two files for validation purpose. In such case, you can create another folder named `test` and place your validation csv files into it. If you have widefield images, you need to export them as .png images (16bit or 8bit grayscale), and then rename then such they will have the same name as the corresponding csv file except the .png extension.
 * Now run the following command to train on your data:

```bash
python run_csv.py --workdir=./training_workdir_exp1
```

With the above code, it will first render histogram images with a subset of the full frame of each csv file. The rendered images will be saved into `__images__`. You need to check the files inside this folder is rendered correctly. You should be able to see images starts with `A_` and `B_`, they will be used as input and target image when training your A-net. If you have your widefield image placed correctly, you should also see images starts with `LR_`. You should check all these images with the help of contrast stretching in ImageJ for example.

When it's done, the training will start. It will produce some interim images in the `outputs` folder which you can used to check whether the training goes well. The model will be saved into `__model__` after a certain epochs of training. Please also notice that If you are running on a cpu, it can be very slow.

## Train with other type of images
You can also use ANNA-PALM to work with other type of images which are not localization table. In such case, follow these steps:
  * Create a folder as your working directory(for example `training_workdir_exp2`) , inside your `training_workdir_exp2`, you need to create a folder named `train` and `test` (optional).
  * Then, place all your images into the `train` folder, optionally place a few images into `test`.

  * Now run the following command as in the above example:
  ```bash
  python run_img.py --workdir=./training_workdir_exp2
  ```

## Monitor your training
If you want to monitor the training progress, you should use the tensorboard interface which provides plots in the browser. In order to launch that, type the following command:
```bash
cd training_workdir_exp1/__model__
tensorboard --logdir=./
```
Then you can open your browser, and go to http://localhost:6006 to see the loss and outputs etc. At the begining, you will only see the A-net graph. As the training goes, you will see a tab with loss curve etc.


## Freeze trained models
Together with ANNA-PALM python code, a ImageJ plugin is provided to

In order to use your trained model in the [imagej plugin](https://s3.eu-west-2.amazonaws.com/anna-palm-model/ANNA_PALM_Process-latest.jar), you need to first train a model, and then run the following script to get a frozen model:
```
python freeze.py --workdir=./frozen_model_sim --load_dir=./results/simulated_model
```
 * use --load_dir to specify the directory contains your trained model
 * use --workdir to specify where you want to save the exported directory, you will find the frozen model file named "tensorflow_model.pb"

Then you can copy the .pb file into ImageJ.
(TODO: how to add frozen model to imagej)

## License
There are two licenses for different part of the ANNA-PALM code: a [`MIT license`](https://github.com/imodpasteur/ANNA-PALM/blob/master/AnetLib/LICENSE) is applied to files inside the `AnetLib` folder. A [`Non-commercial License Agreement`](https://github.com/imodpasteur/ANNA-PALM/blob/master/license.pdf) is applied to all other files.

## FAQ
* How to train faster?
You will need tensorflow compatible GPU, and setup the corresponding drivers and pacakges.

The other option is to reduce the size of the current neural network, for example add the following options in your script:
```
opt.ngf = 16 # this is the base number of filters for the generator, default value: 64
opt.ndf = 16 # this is the base number of filters for the discriminator, default value: 64
```

* If you are using MACOS, and encountered problem with the torch package, try to upgrade to Python3.6.1, for example: by running 'conda install python 3.6.1'

* For more FAQs, check out: https://annapalm.pasteur.fr/#/faq .

* If you can't find an answer, please [contact us](https://oeway.typeform.com/to/qyJOIy) or [add a github issue](https://github.com/imodpasteur/ANNA-PALM/issues).
