#!/bin/bash
# source this file for development purposes - deployment will use wheel files
[[ $_ != $0 ]] && this_is_sourced=1 || this_is_sourced=""
this_script_path="${BASH_SOURCE[@]}"
if [[ ${this_is_sourced} ]]; then
    pushd $(dirname ${this_script_path}) >/dev/null
    this_script_dir=$(pwd)
    popd >/dev/null
    #### set environment vars
    export PYTHONPATH="$this_script_dir"
    export PYTHONDONTWRITEBYTECODE="1"
    /usr/bin/env | grep -Fe "PYTHON" >/dev/tty
    export PATH=${this_script_dir}/bin:$PATH
    ####
    unset this_script_dir
    unset this_script_path
    unset this_is_sourced
else
    echo "usage: source ${this_script_path}"
    exit
fi
