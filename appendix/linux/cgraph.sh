#!/bin/sh

ext=".png"
dotfile=$1
imgfile="${dotfile%.*}"$ext
dot -Gdpi="96" -Tpng $dotfile -o $imgfile
