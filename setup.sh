bash miniconda.sh -b
conda install -yq -c conda-forge --file conda_requirements.txt
pip install -r requirements.txt
python -m spacy download -d en_core_web_sm-1.2.0
python -m spacy link en_core_web_sm en
wget https://storage.googleapis.com/ml-bucket-isikus/cbmodel/err_type_classifier.cbm -P ./models
python setup.py
