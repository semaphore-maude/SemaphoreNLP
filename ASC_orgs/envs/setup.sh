#!/bin/bash
pip install virtualenv
virtualenv py2
p2=`which python2`
virtualenv --no-site-packages --distribute -p $p2 --always-copy py2
virtualenv py3
p3=`which python3`
virtualenv --no-site-packages --distribute -p $p3 --always-copy py3
./update.sh