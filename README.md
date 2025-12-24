# maxwell-code

## steps to build maxone code

This process follows a specific sequence. Begin by compiling the C++ code; execute this step by running the make command from the main folder. This will generate a new directory named build, where the executable file can be found:

``` bash
# cd /path/to/maxlab_lib
make maxone_with_filter
```

The next step will be to run the compiled C++ script:

``` bash
./maxone_with_filter
```

And, finally, we are ready to run the Python setup script, called here pong_setup.py. This can be done with:

``` bash
python3 pong_setup.py
```

Note

To execute the Python script, it is important to install the maxlab package first. If a virtual environment, such as pyenv or conda, is in use, ensure that the environment is activated correctly.

The experiment is in progress, and it can be observed by accessing the GUI. The data is systematically saved.

Other experimental workflows are also possible. Another option is to create a Python setup script that calls the C++ executable from within, so that only the Python script needs to be manually run.
