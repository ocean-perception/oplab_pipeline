#!/bin/bash

# To install it in your system, place this file in your .bashrc or .bash_aliases

function auv_cd {
    if [[ $1 = "--help" ]] || [[ $# -gt 1 ]]; then
        echo -e "usage: auv_cd (raw/configuration/processed)\n\nJump to target dive."
        return 0
    fi
    if [ -z $1 ]; then
        echo -e "usage: auv_cd (raw/configuration/processed)\n\nJump to target dive."
        return 0
    fi

    if [[ $PWD  =~ raw ]]; then
        if [[ $1 == "raw" ]]; then
            echo "You are already at raw"
            return 0
        fi
        echo "Changing directory from raw to "$1
        cd ${PWD/raw/$1}
        return 0
    fi

    if [[ $PWD =~ configuration ]]; then
        if [[ $1 == "configuration" ]]; then
            echo "You are already at configuration"
            return 0
        fi
      echo "Changing directory from configuration to "$1
        cd ${PWD/configuration/$1}
        return 0
    fi

    if [[ $PWD =~ processed ]]; then
        if [[ $1 == "processed" ]]; then
            echo "You are already at processed"
            return 0
        fi
      echo "Changing directory from processed  to "$1
        cd ${PWD/processed/$1}
        return 0
    fi
}
