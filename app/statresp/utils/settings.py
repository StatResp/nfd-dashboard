# -*- coding: utf-8 -*-
"""
@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
"""
import logging
import os
import sys
from argparse import ArgumentParser


#def print_l(msg):
#    Singleton.ins().logger.info(msg)
def print_l(*args):  
    out=''
    #print(args)
    for arg in args: 
        print(arg,end=' ')
        out = out + str(arg)
        Singleton.ins().logger.info(arg)

def print_e(msg):
    Singleton.ins().logger.error(msg)


def print_w(msg):
    Singleton.ins().logger.warning(msg)


class Singleton:
    _instance = None

    def __init__(self):
        self.args = None
        self.output_folder = None


    @classmethod
    def ins(cls):
        if cls._instance is None:
            cls._instance = Singleton()
        return cls._instance

    @classmethod
    def setLogger(cls):
        Singleton.ins().logger = logging.getLogger("LOG")
        Singleton.ins().logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(CustomFormatter())
        Singleton.ins().logger.addHandler(ch)
        file_handler = logging.FileHandler(f'{Singleton.ins().output_folder}/log.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(CustomFileFormatter())
        Singleton.ins().logger.addHandler(file_handler)

    @classmethod
    def configure_settings(cls,output_folder ):
        #print(Input)
        #arg_parser = ArgParser()#Input#ArgParser()
        #Singleton.ins().args = arg_parser.parse_args()
        Singleton.ins().output_folder = output_folder #Singleton.ins().args.output_folder

        if not os.path.exists(Singleton.ins().output_folder):
            os.makedirs(Singleton.ins().output_folder)

        Singleton.ins().setLogger()
        #print_l('starting execution')


    


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    reset = "\x1b[0m"
    _format_ = "%(asctime)s - %(levelname)s - %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    FORMATS = {
        logging.INFO: grey + _format_ + reset,
        logging.WARNING: yellow + _format_ + reset,
        logging.ERROR: red + _format_ + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)


class CustomFileFormatter(logging.Formatter):
    _format_ = "%(asctime)s - %(levelname)s - %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    FORMATS = {
        logging.INFO: _format_,
        logging.WARNING: _format_,
        logging.ERROR: _format_,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)
    
    
class ArgParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, allow_abbrev=True)
        
        #with default
        self.add_argument("-i", "--input_folder", help="input folder to look for the files", default='data',
                          required=False)        
        self.add_argument("-c", "--config", help="configuration file to be read", default='etc/config.conf',
                          required=False)
        self.add_argument("-o", "--output_folder",default='data',
                          help="Output Folder", required=False)
        self.add_argument("-w", "--write",default=False,
                          help="If set True, it rewrites even if the similar output already exists. ", required=False)

        
      
        
class ArgParser_datajoin(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, allow_abbrev=True)
        
        #with default     
        self.add_argument("-c", "--config", help="configuration file to be read", default='etc/config_datajoin.conf',
                          required=False)
        self.add_argument("-o", "--output_folder",default='data',
                          help="Output Folder", required=False)
        self.add_argument("-w", "--write",default=False,
                          help="If set True, it rewrites even if the similar output already exists. ", required=False)


        
