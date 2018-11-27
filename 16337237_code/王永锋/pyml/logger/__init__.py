import logging

# create logger
logger = logging.getLogger('simple_example')
formatter = logging.Formatter('[%(levelname)8s] - [%(module)10s] - [%(lineno)3d] - [%(funcName)10s] \n%(message)s\n')

logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

fl = logging.FileHandler('temp.log',mode='a')
# fl.setLevel(logging.DEBUG)
fl.setFormatter(formatter)
logger.addHandler(fl)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(filename)s - %(module)s - %(funcName)s - %(lineno)d \n%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')