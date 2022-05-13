# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import datetime
import getpass
import logging
import socket
import timeit
from pathlib import Path

import pkg_resources

logger = None  # Public logger
verbose = False


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ""

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print(
            BColors.OKBLUE
            + self.name
            + " took ▸ "
            + BColors.ENDC
            + str(self.took)
            + " ms"
        )


# Singleton class to wrap the console output
class Console:
    """Console utility functions"""

    @staticmethod
    def set_verbosity(verbosity) -> None:
        global verbose
        verbose = verbosity

    @staticmethod
    def warn(*args, **kwargs) -> None:
        """Print a warning message"""
        print(
            BColors.WARNING + "WARN ▸ " + BColors.ENDC + " ".join(map(str, args)),
            **kwargs
        )
        if logger is not None:
            logger.warn(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def warn_verbose(*args, **kwargs) -> None:
        """Print a warning message if in verbose mode. Log in either case."""
        if verbose:
            print(
                BColors.WARNING + "WARN ▸ " + BColors.ENDC + " ".join(map(str, args)),
                **kwargs
            )
        if logger is not None:
            logger.warn(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def error(*args, **kwargs) -> None:
        """Print an error message"""
        print(
            BColors.FAIL + "ERROR ▸ " + BColors.ENDC + " ".join(map(str, args)),
            **kwargs
        )
        if logger is not None:
            logger.error(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def info(*args, **kwargs) -> None:
        """Print an information message"""
        print(
            BColors.OKBLUE + "INFO ▸ " + BColors.ENDC + " ".join(map(str, args)),
            **kwargs
        )
        if logger is not None:
            logger.info(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def info_verbose(*args, **kwargs) -> None:
        """Print an information message if in verbose mode. Log in either case."""
        if verbose:
            print(
                BColors.OKBLUE + "INFO ▸ " + BColors.ENDC + " ".join(map(str, args)),
                **kwargs
            )
        if logger is not None:
            logger.info(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def quit(*args, **kwargs) -> None:
        """Print a FAIL message and stop execution"""
        print("\n")
        print(BColors.FAIL + "**** " + BColors.ENDC + "Exiting.")
        print(
            BColors.FAIL
            + "**** "
            + BColors.ENDC
            + "Reason: "
            + " ".join(map(str, args)),
            **kwargs
        )
        if logger is not None:
            logger.warn(" ".join(map(str, args)), **kwargs)
        quit()

    @staticmethod
    def banner() -> None:
        """Displays Ocean Perception banner and copyright"""
        print(" ")
        print(BColors.OKBLUE + "     ● ● " + BColors.ENDC + " Ocean Perception")
        print(
            BColors.OKBLUE
            + "     ● "
            + BColors.WARNING
            + "▲ "
            + BColors.ENDC
            + " University of Southampton"
        )
        print(" ")
        print(" Copyright (C) 2020 University of Southampton   ")
        print(" This program comes with ABSOLUTELY NO WARRANTY.")
        print(" This is free software, and you are welcome to  ")
        print(" redistribute it.                               ")
        print(" ")

    @staticmethod
    def get_username() -> str:
        """Returns the computer username

        Returns:
            str -- Username
        """
        return getpass.getuser()

    @staticmethod
    def get_hostname():
        """Return the hostname

        Returns:
            str -- Hostname
        """
        return socket.gethostname()

    @staticmethod
    def get_date():
        """Returns current date

        Returns:
            str -- Current date
        """
        return str(datetime.datetime.now())

    @staticmethod
    def get_stamp():
        """Returns current epoch

        Returns:
            str -- Epoch time
        """
        return str(datetime.datetime.now().timestamp())

    @staticmethod
    def get_version(pkg_name="oplab_pipeline"):
        """Returns pkg_name version number

        Parameters
        ----------
        pkg_name : str
            Name of the python package, by default 'oplab_pipeline'

        Returns
        -------
        str
            version number (e.g. "0.1.2")
        """
        return str(pkg_resources.require(pkg_name)[0].version)

    @staticmethod
    def write_metadata():
        """Writes all metadata to a string. Useful to write on processed
        files or configurations.

        Returns:
            str -- String containing computer metadata (username, host, date
            and software version)
        """
        msg = (
            'date: "'
            + Console.get_date()
            + '" \n'
            + 'user: "'
            + Console.get_username()
            + '" \n'
            + 'host: "'
            + Console.get_hostname()
            + '" \n'
            + 'version: "'
            + Console.get_version()
            + '" \n'
        )
        return msg

    @staticmethod
    def progress(
        iteration,
        total,
        prefix="Progress:",
        suffix="Complete",
        length=50,
        decimals=1,
        fill="█",
    ):
        """Call in a loop to create a progress bar in the terminal

        Parameters
        ----------
        iteration : int
            Current iteration
        total : int
            Total number of iterations
        prefix : str
            Prefix string, by default 'Progress:'
        suffix : str
            Suffix string, by default 'Complete'
        length : int
            Character lenght of the progress bar in the console, by default 50
        decimals : int
            Number of decimal places of the percentage, by default 1
        fill : str
            Bar fill character, by default '█'
        """
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
        # Print New Line on Complete
        if iteration >= total - 1:
            print()

    @staticmethod
    def set_logging_file(filename):
        global logger
        folder_path = Path(filename).parent
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        fh = logging.FileHandler(filename)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)  # or any level you want
        logger.addHandler(fh)
        if logger is not None:
            logger.info("oplab_pipeline version: " + str(Console.get_version()))
