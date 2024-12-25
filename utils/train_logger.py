import time
import logging

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
log_index = time.strftime("%m_%d_%H_%M", time.localtime(time.time()))
write_flag = True

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt=DATE_FORMAT,
                    filename='./log/train_log_{}.txt'.format(log_index),
                    filemode='w')


def write_log(s, place='main'):
    global write_flag
    if place == 'network':
        if write_flag:
            write_flag = False
            logging.debug(s)
    else:
        logging.info(s)

