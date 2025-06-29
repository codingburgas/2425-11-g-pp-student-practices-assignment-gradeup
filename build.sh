#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Reset migration history and apply the complete migration
flask db stamp 43975b6cbf11
flask db upgrade 