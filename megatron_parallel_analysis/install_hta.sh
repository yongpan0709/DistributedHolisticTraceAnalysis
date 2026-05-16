#!/bin/bash
HOSTFILE=./hostfile
DHTA_PATH=$1
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

for host in ${hostlist[@]}; do
  echo $host
  ssh -f -n $host "cd $DHTA_PATH; pip install -r requirements.txt;pip install -e ." 
  echo $cmd
  ((COUNT++))
done
