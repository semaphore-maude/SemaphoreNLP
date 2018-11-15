
from datetime import datetime
from os import path
from pathlib import Path
import errno


def time_filename():
    time_file = datetime.now()
    time = '_{}h{}m{}s'.format(time_file.hour,
                                         time_file.minute,
                                         time_file.second)  
    date = '{}-{}-{}'.format(time_file.year,
                             time_file.month,
                             time_file.day) 
    return time, date 


def save_folder_file(save_dir, filename, ext='', optional_folder=''):    
    time, date = time_filename() 
    filename = filename + time + ext                                                 
    dirs = path.join(save_dir, optional_folder, date) 
    full_path = path.join(dirs, filename)
    directory = Path(dirs) 
    
    try:
        if not directory.exists():
            print('Directory does not exist, '\
                  'creating a new directory named {}...\n'.format(dirs))
            directory.mkdir(exist_ok=True, parents=True)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

    return full_path
