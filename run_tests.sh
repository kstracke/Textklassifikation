#!/bin/sh

for file in tests/test*.py
do
	python3 "$file"
done
