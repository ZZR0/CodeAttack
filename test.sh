#!/usr/bin/env bash

num=$#
one=1
zero=0
array=($@) 
while [ ${num} != ${zero} ] 
do
	let "num-=one"
	echo "${array[num]}"
done