#!/bin/bash
python adversarial.py &
AVALANCHE_PID=$!

python watch.py &
WATCH_PID=$!

echo "Press Ctrl+C to stop"
echo "Avalanche PID:" $AVALANCHE_PID 
echo "Watcher PID:" $WATCH_PID
trap "kill $AVALANCHE_PID $WATCH_PID && exit" SIGINT SIGTERM

wait
