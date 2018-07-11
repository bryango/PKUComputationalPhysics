#!/bin/bash
# Zip homework for upload

if [ $# -eq 0 ]
  then
    echo "Please input exN"
    exit 1
fi

git archive HEAD -o .package.tmp/"$1"_release.zip
sleep 1
unzip .package.tmp/"$1"_release.zip -d .package.tmp/
cd .package.tmp/$1/
zip -r "$1"_release.zip ./*
mv "$1"_release.zip ../../
cd ../../
zip -ur "$1"_release.zip assets -x *.git* -x *__pycache__*
/bin/rm -r -v .package.tmp/*
/bin/rm -r -v .package.tmp/.git*
