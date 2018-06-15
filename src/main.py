# -*- coding: utf-8 -*-

"""
 @ main

 created on 20180615
"""

import logging as lg
import sys
import time

lg_level=lg.DEBUG
#
# file_subfix=time.strftime('%Y-%H-%d %h:%m:%s', time.localtime())
# logging.basicConfig(filename=file_subfix+'.log',level=lg_level)
# # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

file_subfix = '../logs/'+time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
file_handler = lg.FileHandler(filename=file_subfix+'.log')
stdout_handler = lg.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

lg.basicConfig(
    level=lg_level,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s -> %(message)s',
    handlers=handlers
)

def test():
    lg()

if __name__ == '__main__':
    lg.debug('af')