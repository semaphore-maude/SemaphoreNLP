#!/bin/bash
source py2/bin/activate
pip freeze > requirements2.txt
deactivate
source py3/bin/activate
pip3 freeze > requirements3.txt
deactivate




