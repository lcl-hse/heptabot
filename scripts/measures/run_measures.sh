echo 'Evaluating heptabot "'$HPT_MODEL_TYPE'" model on architecture "'$MODEL_PLACE'"'
echo
echo "Test 1. Correction task, running time and memory usage"
rm -rf input
rm -rf output
mkdir ./output
mkdir ./input
mkdir ./input/correction
unzip -q ./assets/sample_50ksymbols.zip -d ./input/correction
python prepare_input.py
if [ "$MODEL_PLACE" == "tpu" ]
then
  mv process.pkl ./raw/process_texts.pkl
fi
SECONDS=0
if [ "$MODEL_PLACE" != "tpu" ]
then
  bash colab_execute.sh
else
  source activate heptabot
  python batchify_input.py
  source deactivate 1>/dev/null 2>&1
  bash tpu_run.sh 1>/dev/null 2>&1
  source activate heptabot
  python process_output.py
fi
diff=$SECONDS
RAM_AFTER=$(awk '/MemAvailable/ { printf "%.3f\n", $2/1024/1024 }' /proc/meminfo)
if [ "$MODEL_PLACE" == "gpu" ]
then
  RAM_AFTER=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ printf "%.3f\n", $1/1024 }')
fi
source deactivate 1>/dev/null 2>&1
RAM_USED=$(echo "$RAM_BEFORE $RAM_AFTER" | awk '{print $1-$2}')
echo "RAM used: $RAM_USED GiB"
if [ "$MODEL_PLACE" == "gpu" ]
then
  GPU_RAM_USED=$(echo "$GPU_RAM_BEFORE $GPU_RAM_AFTER" | awk '{print $1-$2}')
  echo "GPU memory used: $GPU_RAM_USED GiB"
fi
echo "Time elapsed: $((($diff / 60) % 60)) minutes $(($diff % 60)) seconds"
tt=$(echo "$diff" | awk '{print $1 / 40.0}')
echo "Average time/text: $tt secs"
ts=$(echo "$diff" | awk '{print $1 / 50.0}')
echo "Average time/symbol: $ts ms"
if [ "$MODEL_PLACE" == "tpu" ]
then
  echo "Note that for TPU RAM usage is not so relevant and elapsed time included system startup unlike in CPU and GPU tests."
fi
echo

echo "Test 2. Competition scores"
jfleg_git_url="https://github.com/keisks/jfleg"
conll14_url="https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz"
m2scorer_url="https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz"
bea19_url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/ABCN.test.bea19.orig"
rm -rf input
rm -rf output
mkdir ./output
mkdir ./input
mkdir ./input/jfleg
mkdir ./input/conll
mkdir ./input/bea
mkdir comp_scores
cd comp_scores
echo "Getting JFLEG from "$jfleg_git_url
git clone -q $jfleg_git_url
cp ./jfleg/test/test.src ../input/jfleg/test.src
echo "Getting CONLL-14 test set from "$conll14_url", M2-scorer from "$m2scorer_url
wget -q $conll14_url -O "conll14st-test-data.tar.gz"
tar -xzf conll14st-test-data.tar.gz
python -c "inf=open('./conll14st-test-data/noalt/official-2014.combined.m2', 'r', encoding='utf-8'); outf=open('../input/conll/official-2014-combined.txt', 'w', encoding='utf-8'); outf.write('\n'.join([sent[2:] for sent in inf.read().split('\n') if sent.startswith('S')])); inf.close(); outf.close()"
wget -q $m2scorer_url -O "m2scorer.tar.gz"
tar -xzf m2scorer.tar.gz
echo "Getting BEA-2019 test set from "$bea19_url
wget -q $bea19_url -O "../input/bea/ABCN.bea19.test.corr"
cd ../
echo "Preparing input for heptabot..."
python prepare_input.py
if [ "$MODEL_PLACE" == "tpu" ]
then
  mv process.pkl ./raw/process_texts.pkl
fi
echo "Processing files..."
if [ "$MODEL_PLACE" != "tpu" ]
then
  bash colab_execute.sh
else
  source activate heptabot
  python batchify_input.py
  source deactivate 1>/dev/null 2>&1
  bash tpu_run.sh 1>/dev/null 2>&1
  source activate heptabot
  python process_output.py
fi
source deactivate 1>/dev/null 2>&1
if [ "$MODEL_PLACE" == "tpu" ]
then
  mv ./raw/process_texts.pkl process.pkl
fi
python prepare_output.py
echo "All files processed."
echo
cd comp_scores
echo "** Testing on CoNLL-2014. See official scorer output below **"
python2 ./m2scorer/scripts/m2scorer.py ./conll.res ./conll14st-test-data/noalt/official-2014.combined.m2
echo
echo "** Testing on JFLEG. See official scorer output below **"
cd jfleg
python -m pip install scipy 1>/dev/null 2>&1
python ./eval/gleu.py -r ./test/test.ref[0-3] -s ./test/test.src --hyp ../jfleg.res
cd ../
echo
echo "** To get BEA scores, please upload our output to the official scoring system. **"
echo "** The system is located at https://competitions.codalab.org/competitions/20229#participate. **"
echo "** Our generated output will start downloading shortly. **"
cp bea.res ABCN.bea19.test.corr
bea_zip_name="bea_test_heptabot_"$HPT_MODEL_TYPE"_"$MODEL_PLACE".zip"
zip -q $bea_zip_name ABCN.bea19.test.corr
cp $bea_zip_name ../
echo
