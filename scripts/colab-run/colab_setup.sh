echo "Initializing virtual environment with python 3.6.9"
mamba create -yq -n heptabot python=3.6.9
source activate heptabot
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
pip install -q transformers==4.1.1
echo

echo "Setting up nltk and spaCy"
python -c 'import nltk; nltk.download("punkt")'
python -m spacy download -d en_core_web_sm-1.2.0
python -m spacy link en_core_web_sm en
pip install -q --upgrade pip
echo

echo "Downloading models"
mkdir models
wget -q --show-progress https://storage.googleapis.com/heptabot/models/external/distilbert_stsb_model.tar.gz -P ./models
tar -xzf ./models/distilbert_stsb_model.tar.gz -C ./models
mkdir ./models/classifier
wget -q --show-progress https://storage.googleapis.com/heptabot/models/classifier/err_type_classifier.cbm -P ./models/classifier
tar -xzf ./models/distilbert_stsb_model.tar.gz -C ./models
wget -q --show-progress https://storage.googleapis.com/heptabot/models/tiny/cpu/t5_tiny_model.tar.gz -P ./models
tar -xzf ./models/t5_tiny_model.tar.gz -C ./models
echo

echo "heptabot is ready to use!"