import logging
import os

def make_sure_path_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)    
    
class default_log(object):
    
      def __init__(self, logfilename='log', log_directory='./log/'):
        self.logfilename = logfilename
        self.log_directory  = log_directory
        make_sure_path_exists( path   = log_directory )
        
        if os.path.exists(log_directory+logfilename+'.log'):
            os.remove(log_directory+logfilename+'.log')
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(filename = log_directory+logfilename+'.log',\
                            level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
      def write(self,line):
        logging.debug(line)

# p1 = Person("John", 36)
# p1.myfunc()
