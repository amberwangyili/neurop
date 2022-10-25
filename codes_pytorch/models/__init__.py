from .model import *
import logging


def build_model(opt):
    if opt['is_init']:  
        m = InitModel(opt)    
    else:
        m = FinetuneModel(opt)
    logger = logging.getLogger('base')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

