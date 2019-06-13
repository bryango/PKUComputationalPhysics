#!/bin/bash
# Zip for release

git archive HEAD -o .package.tmp/"$1"_release.zip
