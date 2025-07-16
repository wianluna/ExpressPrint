#!/usr/bin/env bash
set -e

python3 -m unittest discover tests/ -v -p 'test*.py'
