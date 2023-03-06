#!/bin/bash
result_file=result_less.log
rm -f ${result_file}
files=`ls ../log/*_std_out.log`
for i in ${files[@]}
      do
         echo -e "=========="$i"=========== \n" >>  ${result_file}
         echo -e "\n" >>  ${result_file}
         tail -n 10 $i >>  ${result_file}
         echo -e "\n=====================\n">>  ${result_file}
      done

