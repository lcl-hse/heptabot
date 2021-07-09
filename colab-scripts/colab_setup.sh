echo "Initializing virtual environment with python 3.6.9"
mamba create -q -n heptabot python=3.6.9
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
pip install -q --upgrade pip
echo

echo "Setting up nltk and spaCy"
python -c 'import nltk; nltk.download("punkt")'
python -m spacy download -d en_core_web_sm-1.2.0
python -m spacy link en_core_web_sm en
echo

echo "Downloading models"
mkdir ./models
wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/external/distilbert_stsb_model.tar.gz -P ./models
tar -xzvf ./models/distilbert_stsb_model.tar.gz -C ./models
mkdir ./models/t5-tokenizer
wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/external/sentencepiece.model -P ./models/t5-tokenizer
mv ./models/t5-tokenizer/sentencepiece.model ./models/t5-tokenizer/spiece.model
wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/external/tokenizer.json -P ./models/t5-tokenizer
if [ "$MODEL_PLACE" == "tpu" ]
then
    mkdir ./models/tpumodel
    if [ "$HPT_MODEL_TYPE" == "xxl" ]
    then
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/xxl/tpu/model.ckpt-1014000.data-00000-of-00002 -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/xxl/tpu/model.ckpt-1014000.data-00001-of-00002 -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/xxl/tpu/model.ckpt-1014000.index -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/xxl/tpu/model.ckpt-1014000.meta -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/xxl/tpu/operative_config.gin -P ./models/tpumodel
    else
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/tpu/model.ckpt-1003800.data-00000-of-00002 -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/tpu/model.ckpt-1003800.data-00001-of-00002 -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/tpu/model.ckpt-1003800.index -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/tpu/model.ckpt-1003800.meta -P ./models/tpumodel
        wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/tpu/operative_config.gin -P ./models/tpumodel
    fi
else
    mkdir ./models/savemodel
    wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/gpu/saved_model.pb -P ./models/savemodel
    mkdir ./models/savemodel/variables
    wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/gpu/variables/variables.data-00000-of-00002 -P ./models/savemodel/variables
    wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/gpu/variables/variables.data-00001-of-00002 -P ./models/savemodel/variables
    wget -q --show-progress https://storage.googleapis.com/isikus/heptabot/models/medium/gpu/variables/variables.index -P ./models/savemodel/variables
fi
echo

echo "heptabot is ready to use!"