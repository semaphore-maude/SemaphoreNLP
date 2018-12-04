
from datetime import datetime
from os import path
from pathlib import Path
import errno
import logging

'''
Helper functions for saving info about model
'''


def time_filename():
    '''
    Gives time and date for folder/file as date/filename_time format
    Helps getting precise names to know where the plots/tables come from

    Returns str of date and time
    '''
    time_file = datetime.now()
    time = '_{}h{}m{}s'.format(time_file.hour,
                                         time_file.minute,
                                         time_file.second)  
    date = '{}-{}-{}'.format(time_file.year,
                             time_file.month,
                             time_file.day) 
    return time, date 


def save_folder_file(save_dir, filename, ext='', optional_folder=''):   
    '''
    Creates a string with a precise path to save files with time and dates.
    (using time_filename() fun above)
    Creates new directories if folders given do not exist

    Arguments:
    `save_dir`:        str, name of folder (child of current directory)
    `filename`:        str, a file name without extension. If empty str, creates a filename
    `ext`:             str, file extension (in .ext form)
    `optional_folder`: str, optional sub folder, child of save_dir  

    Returns: str, full path 
    ''' 
    time, date = time_filename() 
    filename = filename + time + ext                                                 
    dirs = path.join(save_dir, optional_folder, date) 
    full_path = path.join(dirs, filename)
    directory = Path(dirs) 
    
    try:
        if not directory.exists():
            print('Directory does not exist, '\
                  'creating a new directory named /{}/...\n'.format(dirs))
            directory.mkdir(exist_ok=True, parents=True)

    except OSError as error: 
        if error.errno != errno.EEXIST:
            raise

    return full_path


def logged(save_dir='log', filename=''):
    '''
       Produces a log file. Creates a directory if save_dir does not exists
     * Required * for convergence plot (loglik vs iter) with LDA.

     Arguments:
      `save_dir`:        str, name of folder (child of current directory)
      `filename`:        str, a file name without extension. If empty str, creates a filename
     Returns: file path to use for plotting
    '''
    full_path_log = save_folder_file(save_dir=save_dir, filename=filename, ext='.log')
    
    logging.basicConfig(filename=full_path_log, 
                      format='%(asctime)s : %(levelname)s : %(message)s', 
                      level=logging.INFO)  

    return full_path_log



    