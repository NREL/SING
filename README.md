# What is SING and who is it for?

Synthetic dIstribution Network Generator is a standalone python module that is able to create synthetic distribution models for OpenDSS using GIS datasets. The software uses road and building information from OpenStreetMaps to generate these synthetic models. The software uses Bokeh platform and the simple to use dashboard allows users to build rich synthetic distribution models down to individual loads. The supported dashboard allows users to tweek a handful of parameters that inform the machine learning \ heuristic methods employed to build the network.  

# Installing SING

SING installation requires Python Anaconda,.

The repository can be cloned using the following command within a command prompt window

- git clone https://github.nrel.gov/alatif/SING.git

A new conda environment can be created using the following command

- conda create --name <env> --file requirements.txt

Finally, the module can be installed using the command within the cloned repo directory

- pip install -e. 


# Running SING

The SING app can be launched using the following command within the cloned repo directory

- bokeh serve model.py