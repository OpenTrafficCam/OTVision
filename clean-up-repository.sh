#!/bin/bash

git checkout main || git checkout master
git fetch --all --prune
git gc
git branch -vv | grep -E ': gone]|: entfernt]' |  grep -v "\*" | awk '{ print $1; }' | xargs -r git branch -d
