#!/bin/bash

case $HOSTNAME in
  (wallenstein) chunk=0;;
  (april)   chunk=1;;
  (february) chunk=2;;
  (caesar) chunk=3;;
  (attilla) chunk=4;;
  (odoaker) chunk=5;;
  (priamos) chunk=6;;
  (caligula) chunk=7;;
  (*) chunk=8;;
  #(orpheus) chunk=1;;
  #(ilos) chunk=4;;
  #(otho) chunk=4;;
esac

source ${HOME}/code/he/devel/setup.bash

Xvfb :8 &
export Xvfb_PID=$!
DISPLAY=:8 roslaunch haptic_exploration record_ycb_data.launch num_splits:=7 chunk:=$chunk
kill $Xvfb_PID
