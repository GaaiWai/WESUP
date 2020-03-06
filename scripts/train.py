"""
Training module.
"""

import logging
from shutil import rmtree

import fire

from models import initialize_trainer
from utils.metrics import accuracy
from utils.metrics import dice


def fit(dataset_path, model='mild', **kwargs):
    # Initialize logger.
    logger = logging.getLogger('Train')#初始化logging
    logger.setLevel(logging.DEBUG)#将日志等级设置为DEBUG
    logger.addHandler(logging.StreamHandler())#日志输出必须有一个handler，将StreamHandler作为logger的句柄
    print('train_basic')
    trainer = initialize_trainer(model, logger=logger, **kwargs)#初始化trainer

    try:
        trainer.train(dataset_path,
                      metrics=[accuracy, dice], **kwargs)
    finally:
        if kwargs.get('smoke'):
            rmtree(trainer.record_dir, ignore_errors=True)


if __name__ == '__main__':
    fire.Fire(fit)
