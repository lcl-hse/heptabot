echo "heptabot is starting and will be ready in 75 seconds"
tryenv=`conda activate heptabot`
if [[ $tryenv != "" ]]
then
    source ~/mambaforge/etc/profile.d/conda.sh
    conda activate heptabot
fi
pyro4-ns &
sleep 5; python models.py &
sleep 70; python main.py &