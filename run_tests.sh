#!/bin/sh

for file in tests/test*.py
do
	python "$file"
done
