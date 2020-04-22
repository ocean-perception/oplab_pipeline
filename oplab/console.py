import socket
import getpass
import datetime
import pkg_resources


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Singleton class to wrap the console output
class Console:
    @staticmethod
    def warn(*args, **kwargs):
        print(BColors.WARNING + "WARN ▸ " + BColors.ENDC + " ".join(map(str, args)), **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        print(BColors.FAIL + "ERROR ▸ " + BColors.ENDC + " ".join(map(str, args)), **kwargs)

    @staticmethod
    def info(*args, **kwargs):
        print(BColors.OKBLUE + "INFO ▸ " + BColors.ENDC + " ".join(map(str, args)), **kwargs)

    @staticmethod
    def quit(*args, **kwargs):
        print('\n')
        print(BColors.FAIL + "**** " + BColors.ENDC + "Exitting auv_nav.")
        print(BColors.FAIL + "**** " + BColors.ENDC + "Reason: " + " ".join(map(str, args)), **kwargs)
        quit()

    @staticmethod
    def banner():
        print(' ')
        print(BColors.OKBLUE + '     ● ● ' + BColors.ENDC + ' Ocean Perception')
        print(BColors.OKBLUE + '     ● ' + BColors.WARNING + '▲ ' + BColors.ENDC + ' University of Southampton')
        print(' ')
        print(' Copyright (C) 2020 University of Southampton   ')
        print(' This program comes with ABSOLUTELY NO WARRANTY.')
        print(' This is free software, and you are welcome to  ')
        print(' redistribute it.                               ')
        print(' ')

    @staticmethod
    def get_username():
        return getpass.getuser()

    @staticmethod
    def get_hostname():
        return socket.gethostname()

    @staticmethod
    def get_date():
        return str(datetime.datetime.now())

    @staticmethod
    def get_stamp():
        return str(datetime.datetime.now().timestamp())

    @staticmethod
    def get_version():
        return str(pkg_resources.require("auv_nav")[0].version)

    @staticmethod
    def progress(iteration, total, prefix='Progress:', suffix='Complete',
                 length=50, decimals=1, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            length      - Optional  : character length of bar (Int)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration >= total - 1:
            print()
