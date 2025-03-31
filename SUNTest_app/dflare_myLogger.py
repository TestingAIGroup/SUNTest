import os
import logging


def create_folder(path, safe=True):
    if not os.path.exists(path):
        print("Creating Dirs: {}".format(path))
        os.makedirs(path)
    else:
        print("Dirs: {} Exists".format(path))
        if safe:
            assert False, "Dir exist: {}".format(path)


def create_logger(filepath, logger_name=None):
    if logger_name is None:
        logger_name = filepath

    create_folder(os.path.dirname(filepath), False)
    logging.basicConfig(filename=filepath, filemode='a', level=logging.INFO, datefmt='%m-%d %H:%M:%S',
                        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(logger_name)
    # logger.addHandler(logging.FileHandler(filepath, 'a'))
    log = logger.info
    return log
