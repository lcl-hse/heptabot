echo "This script needs conda, so it will install miniconda if conda is not found."
echo "It requires wget and Python 3."
echo "It will also initialize a new virtual environment and download all the necessary files."
read -p "Enter 'y' to proceed with heptabot setup: " -n 1 -r
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Terminating on user request"
	[[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi
echo

echo "Checking conda installation"
cv=`which conda`
if [[ $cv == "" ]]
then
    echo "conda not foung, installing"
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
fi
echo

echo "Initializing virtual environment with python 3.6.9"
conda install nb_conda
conda create -n heptabot python=3.6.9
conda activate heptabot

pyv="$(python -V 2>&1)"
if [[ $pyv != "3.6.9" ]]
then
    echo "Python version is not 3.6.9, exiting"
    exit 3
fi
echo

echo "Installing requirements"
conda install -yq -c conda-forge --file conda_requirements.txt
pip install -r requirements.txt
echo

echo "Setting up nltk and spaCy"
python -c 'import nltk; nltk.download("punkt")'.
python -m spacy download -d en_core_web_sm-1.2.0
python -m spacy link en_core_web_sm en
echo

echo "Downloading models"
wget -q --show-progress https://storage.googleapis.com/ml-bucket-isikus/cbmodel/err_type_classifier.cbm -P ./models
mkdir ./models/savemodel
wget -q --show-progress https://storage.googleapis.com/ml-bucket-isikus/t5-base-model/models/base-basedrei/export/1599625548/saved_model.pb -P ./models/savemodel
mkdir ./models/savemodel/variables
wget -q --show-progress https://storage.googleapis.com/ml-bucket-isikus/t5-base-model/models/base-basedrei/export/1599625548/variables/variables.data-00000-of-00002 -P ./models/savemodel/variables
wget -q --show-progress https://storage.googleapis.com/ml-bucket-isikus/t5-base-model/models/base-basedrei/export/1599625548/variables/variables.data-00001-of-00002 -P ./models/savemodel/variables
wget -q --show-progress https://storage.googleapis.com/ml-bucket-isikus/t5-base-model/models/base-basedrei/export/1599625548/variables/variables.index -P ./models/savemodel/variables
echo

echo "heptabot is ready to use!"
echo "run python main.py to activate the system"
exit 0