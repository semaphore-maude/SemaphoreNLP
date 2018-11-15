## instructions for running the shell scripts to open new environments

#### * Note: all the scripts are meant to be run -outside- the environments *

#### The way we can make sure we have the same environments is by *committing* the requirements#.txt files

#### - install python 2 if needed
#### - install python 3 if needed

### from this repo:

    git pull
    
    ./setup.sh

### to start python(#) environment, where # is 2 or 3 

    source py#/bin/activate

### when done: 

    deactivate

### after an update (when someone has committed a change to git and you want to make it active in your environments):

#### - make sure you're deactivated
    
    git pull 

    ./update.sh

### after adding a new package:

#### - make sure you're deactivated

    ./freeze.sh
    
    # requirements#.txt - commit whichever of 2 or 3 is changed
    
    git add requirements#.txt
    git commit -m "yolo=true 😎"
    git push
       
    
    
    
