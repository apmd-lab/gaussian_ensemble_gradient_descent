## Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate

## Install Packaging Tools
pip install --upgrade pip
pip install setuptools wheel twine numpy auditwheel # numpy is optional

## Generate Distribution Files
python setup.py sdist bdist_wheel

## Upload package
twine upload dist/*

## Exit Virtual Environment
deactivate

## Example Project Structure
## my_project/
## 戍式式 setup.py
## 戍式式 README.md
## 戍式式 LICENSE
## 戍式式 MANIFEST.in
## 戌式式 eschallot/
##     戍式式 __init__.py
##     戍式式 mie/
##     弛   戍式式 __init__.py
##     弛   戌式式 module.py
##     戌式式 optimization/
##         戍式式 __init__.py
##         戌式式 module.py

## Repair Wheels if not Pure Python
# auditwheel repair dist/*linux_x86_64.whl -w dist/
# twine upload dist/*manylinux*.whl dist/*.tar.gz