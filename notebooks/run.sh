#!/bin/bash

for index in $(seq 0 23); do
		echo ${index}
		jupyter nbconvert --to notebook --execute Figure5.ipynb  --stdout > /dev/null --ExecutePreprocessor.extra_arguments=$index
done
