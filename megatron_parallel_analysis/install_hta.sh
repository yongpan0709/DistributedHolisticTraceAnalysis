#!/bin/bash
HOSTFILE=./hostfile
HTA_PATH=$1
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

for host in ${hostlist[@]}; do
  echo $host
  ssh -f -n $host "cd $HTA_PATH; pip install -r requirements.txt;pip install -e ." 
  # ssh -f -n $host "apt update; apt install -y tmux"  
  echo $cmd
  ((COUNT++))
done
