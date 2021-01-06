#!/bin/sh

ENV_DIR='.venv_py2'

if [[ ! -d $ENV_DIR ]]
then
    echo "Installing required packages."
    /usr/bin/virtualenv $ENV_DIR
    source $ENV_DIR/bin/activate
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "Activating environment."
    source $ENV_DIR/bin/activate
fi

export PYTHONPATH='.'
