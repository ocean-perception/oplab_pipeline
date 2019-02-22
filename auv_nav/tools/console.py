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
