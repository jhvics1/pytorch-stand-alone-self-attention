#!/usr/bin/env bash
DIR="train"
for fname in $DIR/*.tar; do
    echo $fname
    dst="${fname/.tar/}"
    mkdir -p $dst
    tar xf $fname --directory $dst
    rm $fname
done
