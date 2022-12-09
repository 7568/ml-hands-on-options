#!/bin/bash
files=`ls pid`
for i in ${files[@]}
      do
          while IFS= read -r line;
          do
            kill -9 $line
          done < 'pid'/$i
      done

rm -rf pid/*
rm -rf ../log/*