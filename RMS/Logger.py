import os
import sys
import errno
import logging
import logging.handlers
import multiprocessing
import datetime
import time

# Set GStreamer debug level. Use '2' for warnings in production environments.
os.environ['GST_DEBUG'] = '2'


##############################################################################
# GLOBALS
##############################################################################
log_queue = None
listener_process = None
logger_initialized = False


##############################################################################
# HELPERS
##############################################################################
class LoggerWriter:
    """Used to redirect stdout/stderr to the log."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

def mkdirP(path):
    """Makes a directory and handles all errors."""
    try:
        os.makedirs(path)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return True
        else:
            print("Error creating directory: " + str(exc))
            return False
    except Exception as e:
        print("Error creating directory: " + str(e))
        return False
    return False


class RmsDateTime:
    """Use Python-version-specific UTC retrieval."""
    if sys.version_info[0] < 3:
        @staticmethod
        def utcnow():
            return datetime.datetime.utcnow()
    else:
        @staticmethod
        def utcnow():
            return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


#############################################################################
# CUSTOM TIMEDROTATINGFILEHANDLER CLASS
#############################################################################
class CustomHandler(logging.handlers.TimedRotatingFileHandler):
    """
    - The live file: log_US005A_2024-12-29_112347.log
    - On rollover, rename to: log_US005A_2024-12-29_112347-[29_1123-to-30_1123].log
    """
    def __init__(self, station_id, start_time_str, *args, **kwargs):
        self.station_id = station_id
        self.start_time_str = start_time_str
        super(CustomHandler, self).__init__(*args, **kwargs)
        self.suffix = "%Y-%m-%d_%H%M%S"
        self.namer = self._rename_on_rollover

    def _rename_on_rollover(self, default_name):
        base_dir, base_file = os.path.split(default_name)
        base_noext, dot, start_time_str = base_file.rpartition('.')
        
        if base_noext.endswith('.log'):
            base_noext = base_noext[:-4]
        
        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d_%H%M%S")
        end_time = datetime.datetime.fromtimestamp(self.rolloverAt)
        
        start_str = start_time.strftime("%d_%H%M")
        end_str = end_time.strftime("%d_%H%M")
        
        new_name = "{}-[{}-to-{}].log".format(base_noext, start_str, end_str)
        return os.path.join(base_dir, new_name)


##############################################################################
# LISTENER SIDE
##############################################################################
class NoiseFilter(logging.Filter):
    """Filter out noisy messages from specific modules."""
    def __init__(self):
        super(NoiseFilter, self).__init__()
        self.noisy_modules = {'font_manager', 'ticker', 'transport', 'sftp', 'dvrip', 'channel', 'cmd'}

    def filter(self, record):
        if record.levelno in (logging.DEBUG, logging.INFO) and record.module in self.noisy_modules:
            return False
        return True

def _listener_configurer(config, log_file_prefix, safedir):
    """
    Set up the root logger with a TimedRotatingFileHandler. 
    This runs in the separate listener process.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Build the path for logs
    log_path = os.path.join(config.data_dir, config.log_dir)
    mkdirP(log_path)

    # If not writable, use safedir if given
    if safedir:
        if not os.path.exists(log_path) or not os.access(log_path, os.W_OK):
            root_logger.debug("Log directory not writable, using safedir: %s", safedir)
            log_path = safedir
            mkdirP(log_path)

    # Generate filename with prefix, station ID, timestamp
    start_time_str = RmsDateTime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    logfile_name = "{}log_{}_{}.log".format(log_file_prefix, config.stationID, start_time_str)
    full_path = os.path.join(log_path, logfile_name)

    # Set up rotating handler
    handler = CustomHandler(
        station_id=config.stationID,
        start_time_str=start_time_str,
        filename=full_path,
        when='H',
        interval=12,
        utc=True
    )
    handler.setLevel(logging.DEBUG)
    handler.addFilter(NoiseFilter())

    # Set formatter
    formatter = logging.Formatter(
        # fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s',
        fmt='%(asctime)s-%(levelname)s-%(module)s-line: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Console output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    console.addFilter(NoiseFilter())

    # Attach handlers
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.addHandler(console)
    root_logger.propagate = False
    root_logger.debug("Log listener configured. Current file: %s", full_path)


def _listener_process(queue, config, log_file_prefix, safedir):
    """
    Target function for our separate logging listener process.
    Ignores SIGINT and runs a QueueListener so the main process 
    can log asynchronously via the queue.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _listener_configurer(config, log_file_prefix, safedir)

    root_logger = logging.getLogger()
    queue_listener = logging.handlers.QueueListener(queue, *root_logger.handlers)
    queue_listener.start()
    
    while True:
        time.sleep(60)  # keep process alive


##############################################################################
# PUBLIC ENTRY POINT
##############################################################################

def initLogging(config, log_file_prefix="", safedir=None):
    """
    Called once in the MAIN process (e.g. StartCapture.py). 
    Spawns the listener process that owns the TimedRotatingFileHandler.
    All logs from main/child scripts will be funneled through a QueueHandler.

    Arguments:
        config           - RMS config object with .data_dir, .log_dir, .stationID, .log_stdout, etc.
        log_file_prefix  - Optional string prefix for log filenames
        safedir          - Fallback directory if the normal log_path is unwritable
    """
    global log_queue, listener_process, logger_initialized
    if logger_initialized:
        # Already done; do nothing
        return

    # 1) Create a global queue
    log_queue = multiprocessing.Queue(-1)

    # 2) Spawn the listener
    listener_process = multiprocessing.Process(
        target=_listener_process,
        args=(log_queue, config, log_file_prefix, safedir),
        daemon=True
    )
    listener_process.start()

    # 3) In the main process, attach a QueueHandler to the root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)
    # For Additional debugging uncomment this line and comment out the next
    # qh.setFormatter(logging.Formatter('[%(processName)s] %(message)s'))
    qh.setFormatter(logging.Formatter('%(message)s'))

    # Clear any existing handlers in the main process:
    root.handlers = []
    root.addHandler(qh)

    # Always capture stderr
    sys.stderr = LoggerWriter(root, logging.WARNING)

    # Optionally capture stdout
    if config.log_stdout:
        sys.stdout = LoggerWriter(root, logging.INFO)

    root.propagate = False
    logger_initialized = True
    root.debug("initLogging completed; queue listener started.")

def getLogger(name=None):
    return logging.getLogger(name if name else "logger")