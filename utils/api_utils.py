import os
import logging
import datetime

def extract_args_kwargs(command,not_passed_token="not_passed"):
    args = [arg.split('=')[0] for arg in command if len(arg.split('=')) == 1]
    args = args[1:]
    kwargs = dict()
    for arg in command:
        key_value = arg.split('=')
        if len(key_value) == 2 and key_value[1] != not_passed_token:
            kwargs |= {key_value[0]:key_value[1]}
    return args,kwargs

def build_logging(app_name,
                  log_date=datetime.datetime.utcnow(),
                  log_mapping=None):
    '''log_mapping={logging.DEBUG:'debug.log'...
    3 handlers: >=debug, ==info and >=warning
    if not log_date no date in filename'''

    class MyFilter(object):
        '''this is to restrict info logger to info only'''
        def __init__(self, level):
            self.__level = level
        def filter(self, logRecord):
            return logRecord.levelno <= self.__level

    # mkdir log repos if does not exist
    log_path = os.path.join(os.sep, os.getcwd(), "logs", app_name)
    if not os.path.exists(log_path):
        os.umask(0)
        os.makedirs(log_path, mode=0o777)

    logging.basicConfig()
    logger = logging.getLogger(app_name)

    # logs
    if log_mapping is None:
        log_mapping = {logging.INFO:'info.log',logging.WARNING:'warning.log',logging.CRITICAL:'program_flow.log'}
    for level,filename in log_mapping.items():
        handler = logging.FileHandler(os.path.join(os.sep,log_path,f'{log_date.strftime("%Y%m%d_%H%M%S")}_{filename}' if log_date else filename), mode='w')
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
        #handler.addFilter(MyFilter(level))
        logger.addHandler(handler)

    # handler_alert = logging.handlers.SMTPHandler(mailhost='smtp.google.com',
    #                                              fromaddr='david@pronoia.link',
    #                                              toaddrs=['david@pronoia.link'],
    #                                              subject='auto alert',
    #                                              credentials=('david@pronoia.link', ''),
    #                                              secure=None)
    # handler_alert.setLevel(logging.CRITICAL)
    # handler_alert.setFormatter(logging.Formatter(f"%(levelname)s: %(message)s"))
    # self.myLogger.addHandler(handler_alert)

    logger.setLevel(min(log_mapping.keys()))

    return logger