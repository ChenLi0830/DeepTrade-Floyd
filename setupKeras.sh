#!/bin/bash
cp keras_used/losses.py /usr/local/lib/python3.6/site-packages/keras/losses.py
cp keras_used/activations.py /usr/local/lib/python3.6/site-packages/keras/activations.py
echo "Finished replacing Keras function files"
