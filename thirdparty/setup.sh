#!/bin/bash

if [ ! -r thirdparty ] ; then
  echo "run this script from the root of the repo"
  exit 1
fi

if [ ! -r squidasm ] ; then
  ret=0
  if [ "$NETSQUIDPYPI_USER" == "" ] ; then
    echo "you must define environment variable NETSQUIDPYPI_USER"
    ret=1
  fi
  if [ "$NETSQUIDPYPI_PWD" == "" ] ; then
    echo "you must define environment variable NETSQUIDPYPI_PWD"
    ret=1
  fi
  if [ "$(which virtualenv)" == "" ] ; then
    echo "you must have virtualenv installed"
    ret=1
  fi
  if [ $ret -ne 0 ] ; then
    exit $ret
  fi

  virtualenv .venv -p python3
  source .venv/bin/activate
  pushd thirdparty
  git clone https://github.com/QuTech-Delft/squidasm.git
  pushd squidasm
  make install && make verify
  popd
  popd
fi
