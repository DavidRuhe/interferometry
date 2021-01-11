#!/bin/sh

ENV_DIR='.venv_py3'

if [[ ! -d $ENV_DIR ]]
then
    python3 -m venv $ENV_DIR
    source $ENV_DIR/bin/activate
    # pip install -r requirements_py3.txt -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "Activating environment."
    source $ENV_DIR/bin/activate
fi

# DIFF=$(comm -23 <(sort requirements_py3.txt) <(pip freeze | sort| sed s/=.*//))
# pip install $DIFF -f https://download.pytorch.org/whl/torch_stable.html

pip -q install -r requirements_py3.txt -f https://download.pytorch.org/whl/torch_stable.html

cd src/
export PYTHONPATH=$(pwd)
export JUPYTER_PATH=$(pwd)
