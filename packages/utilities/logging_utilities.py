import logging

global_log_file = './default.log'
global_log_level = logging.INFO
global_format_message = '%(levelname)s:%(name)s:%(message)s'

def global_config_logger(log_file = None, log_level = None, format_message = None):
    """Function to define logger's properties.

    :param log_file: Path to store log file, defaults to None
    :type log_file: str, optional
    :param log_level: Flag to determine logging level, defaults to None
    :type log_level: int, optional
    :param format_message: Format style of log messages, defaults to None
    :type format_message: str, optional
    """
    if log_file:
        global global_log_file
        global_log_file = log_file

    if log_level:
        global global_log_level
        global_log_level = log_level

    if format_message:
        global global_format_message
        global_format_message = format_message

def setup_logger(name):
    """Function to setup the logger with configurations

    :param name: Name of the file using the logger
    :type name: str
    :return: Logger object to be used
    :rtype: Logger
    """
    handler = logging.FileHandler(global_log_file)
    formatter = logging.Formatter(global_format_message)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(global_log_level)
    logger.addHandler(handler)

    return logger

def update_log_level(log_level):
    """Function to update the logging level with a new level.

    :param log_level: Flag to determine the new logging level
    :type log_level: int
    """
    if log_level:
        global global_log_level
        global_log_level = log_level

# https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
def addLoggingLevel(levelName, levelNum, methodName=None):
    """Comprehensively adds a new logging level to the `logging` module and the currently configured logging class.

    :param levelName: Attibute name of the new log level
    :type levelName: str
    :param levelNum: Flag value to use for logging
    :type levelNum: int
    :param methodName: How to call the new log method. Defaults to levelName.lower()
    :type methodName: str, optional
    :raises AttributeError: Logging module name already defined.
    :raises AttributeError: Logging level already defined.
    :raises AttributeError: Logging method already defined.
    """
    if not methodName:
        methodName = levelName.lower()

    # if hasattr(logging, levelName):
    #    raise AttributeError('{} already defined in logging module'.format(levelName))
    # if hasattr(logging, methodName):
    #    raise AttributeError('{} already defined in logging module'.format(methodName))
    # if hasattr(logging.getLoggerClass(), methodName):
    #    raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        """Helper logger function

        :param message: Logging message
        :type message: str
        """
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        """Helper logger function

        :param message: Logging message
        :type message: str
        """
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)