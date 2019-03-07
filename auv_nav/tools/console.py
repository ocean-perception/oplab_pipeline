import socket
import getpass
import datetime
import pkg_resources


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Console:
    def warn(msg):
        print(bcolors.WARNING + "[WARN]: " + bcolors.ENDC + msg)

    def error(msg):
        print(bcolors.FAIL + "[ERROR]: " + bcolors.ENDC + msg)

    def info(msg):
        print(bcolors.OKBLUE + "[INFO]: " + bcolors.ENDC + msg)

    def quit(msg):
        print('\n')
        print(bcolors.FAIL + "[****]: " + bcolors.ENDC + "Exitting auv_nav.")
        print(bcolors.FAIL + "[****]: " + bcolors.ENDC + "Reason: " + msg)
        quit()

    def get_username():
        return getpass.getuser()

    def get_hostname():
        return socket.gethostname()

    def get_date():
        return datetime.datetime.now()

    def get_version():
        return pkg_resources.require("auv_nav")[0].version

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
