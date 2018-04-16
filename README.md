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
 * pytorch==0.2.0.post3
 * torchvision
 * imreg_dft

The recommended way to get python3.6 is to use ANACONDA, go the website: https://www.continuum.io/downloads ,
and download Anaconda(Python 3.6), then install it.

The next step is to install the dependencies, mainly for tensorflow.
Open a terminal, switch to the annaPALM source code folder.

For a quick test running on CPU, run the following command:
```
pip install -r requirements.txt
```

Or if you have an tensorflow compatible GPU, and you want to use GPU for training, run the following command:
```
pip install -r requirements-gpu.txt
```


## USAGE

Two type of scripts are available, the first one is to run on simulated microtubules and nuclear pores.

For example, you can train a new ANNA-PALM model with the following command:
```
python run.py --workdir=./results/simulated_model
```

You can run the following command to see all the arguments.
```
# for example
python3 run.py --help
```

## Train with your own data
Coming Soon...


## Freeze trained model
In order to use your trained model in our imagej plugin, you need to run the following script
```
python freeze.py --workdir=./frozen_model_sim --load_dir=./results/simulated_model
```
 * use --load_dir to specify the directory contains your trained model
 * use --workdir to specify where you want to save the exported directory, you will find the frozen model file named "tensorflow_model.pb"

Frozen models can be loaded with our ImageJ plugin. (TODO: how to add frozen model to imagej)

## FAQ

* If you are using MACOS, and encountered problem with the torch package, try to upgrade to Python3.6.1, for example: by running 'conda install python 3.6.1'

* For more FAQs, check out: https://annapalm.pasteur.fr/#/faq .

* If you can't find an answer, please [contact us](https://oeway.typeform.com/to/qyJOIy) or [add a github issue](https://github.com/imodpasteur/ANNA-PALM/issues).
