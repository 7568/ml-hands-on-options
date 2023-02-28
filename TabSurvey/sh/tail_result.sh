#!/bin/bash
rm -f result.txt
files=`ls ../log/*_std_out.log`
for i in ${files[@]}
      do
         echo -e "===================== \n" >> result.txt
         echo $i >> result.txt
         echo -e "\n" >> result.txt
         tail -n 10 $i >> result.txt
         echo -e "\n=====================\n">> result.txt
      done

