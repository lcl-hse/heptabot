source activate heptabot
echo "heptabot is starting and will be ready in 75 seconds"
pyro4-ns &
sleep 5; python models.py &
sleep 70; python main.py &