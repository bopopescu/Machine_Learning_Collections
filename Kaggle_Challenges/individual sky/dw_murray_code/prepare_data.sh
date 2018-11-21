#!/bin/bash

set -e

for a in Training_halos.csv Train_Skies/Training_Sky{1..300}.csv Test_Skies/Test_Sky{1..120}.csv ; do
    if [ \! -e data/"$a" ] ; then
        echo
        echo Missing file ./data/"$a"
        echo "Please get the test/train .csv files from the kaggle site,"
        echo "unpack and put in the ./data directory"
        echo
        exit 1
    fi
done

BASE=$(dirname "$0")
cd "$BASE"
cd simple_data
python munge1.py
python munge2.py
python tmunge1.py
python tmunge2.py

