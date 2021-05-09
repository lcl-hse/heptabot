echo "This script sets up a new 'heptabot' virtual environment and downloads all the necessary files."
echo "It depends on mamba, git and wget."
echo "We strongly suggest following the https://github.com/lcl-hse/heptabot/blob/tensorflow/notebooks/Install.ipynb notebook to avoid any unexpected problems."

echo "Initializing virtual environment with python 3.6.9"
mamba install nb_conda -yq -c conda-forge
mamba create -q -n heptabot python=3.6.9
source ~/mambaforge/etc/profile.d/conda.sh
conda activate heptabot
pip install -q --upgrade pip

pyv="$(python -V 2>&1)"
if [[ $pyv != *"3.6.9"* ]]
then
    echo "Python version is not 3.6.9, exiting"
    exit 3
fi
echo

echo "Installing requirements"
mamba install -yq -c conda-forge --file conda_requirements.txt
pip install -q -r requirements.txt
pip install -q --upgrade pip
echo

echo "Setting up nltk and spaCy"
python -c 'import nltk; nltk.download("punkt")'
python -m spacy download -d en_core_web_sm-1.2.0
python -m spacy link en_core_web_sm en
echo

echo "Setting up heptabot for jupyter (optional)"
python -m ipykernel install --user --name=heptabot
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
echo "run conda init; conda activate heptabot; ./start.sh to activate the system"
exit 0
