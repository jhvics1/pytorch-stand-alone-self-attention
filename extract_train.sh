#!/usr/bin/env bash
DIR="train"
for fname in $DIR/*.tar; do
    echo $fname
    tar xf $fname --directory $DIR
    rm $fname
done
