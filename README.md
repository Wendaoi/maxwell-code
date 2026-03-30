# maxwell-code

## steps to build maxone code

This process follows a specific sequence. Begin by compiling the C++ code; execute this step by running the make command from the main folder. This will generate a new directory named build, where the executable file can be found:

``` bash
# cd /path/to/maxlab_lib
make maxone_with_filter
```

The next step will be to run the compiled C++ script:

``` bash
./maxone_with_filter /path/to/runtime_config.json
```

And, finally, we are ready to run the Python setup script. For the paper-style cartpole closed-loop workflow, use `cartpole_setup.py`:

``` bash
python3 closedloop/cartpole_setup.py --duration 15 --mode cycled_adaptive
```

Note

To execute the Python script, it is important to install the maxlab package first. If a virtual environment, such as pyenv or conda, is in use, ensure that the environment is activated correctly.

Use `--show-gui` if you want the Qt cartpole viewer while the experiment is running. The runtime config and episode log are written to `~/cartpole_experiments`.

Other experimental workflows are also possible. Another option is to create a Python setup script that calls the C++ executable from within, so that only the Python script needs to be manually run.
