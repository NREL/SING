# What is SING and who is it for?

Synthetic dIstribution Network Generator is a standalone python module that is able to create synthetic distribution models for OpenDSS using GIS datasets. The software uses road and building information from OpenStreetMaps to generate these synthetic models. The software uses Bokeh platform and the simple to use dashboard allows users to build rich synthetic distribution models down to individual loads. The supported dashboard allows users to tweek a handful of parameters that inform the machine learning \ heuristic methods employed to build the network.  

# Installing SING

SING installation requires Python Anaconda,.

The repository can be cloned using the following command within a command prompt window

- git clone https://github.com/NREL/SING.git

A new conda environment can be created using the following command

- conda create --name <env> --file requirements.txt

Finally, the module can be installed using the command within the cloned repo directory

- pip install -e. 


# Running SING

The SING app can be launched using the following command within the cloned repo directory

- bokeh serve model.py

# License

BSD 3-Clause License

Copyright (c) 2022 Alliance for Sustainable Energy, LLC, All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

- Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


