#!/usr/bin/env bash

rm submission.*
zip -j submission.zip Makefile main.sh main.py models_selected.pickle train.csv
