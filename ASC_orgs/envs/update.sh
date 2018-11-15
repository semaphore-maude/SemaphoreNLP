#!/bin/bash
source py2/bin/activate
pip install -r requirements2.txt
deactivate
source py3/bin/activate
pip3 install -r requirements3.txt
deactivate