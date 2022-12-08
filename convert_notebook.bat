echo off
set notebook_name=%1
shift
shift
python -m nbconvert --to webpdf %notebook_name%
