#!/bin/bash
cd ..
for i in * ; do
  if [ -d "$i" ]; then
    du -sh $i
    find $i -type f | wc -l
  fi
done
