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
## ������ setup.py
## ������ README.md
## ������ LICENSE
## ������ MANIFEST.in
## ������ eschallot/
##     ������ __init__.py
##     ������ mie/
##     ��   ������ __init__.py
##     ��   ������ module.py
##     ������ optimization/
##         ������ __init__.py
##         ������ module.py

## Repair Wheels if not Pure Python
# auditwheel repair dist/*0.2.11*linux_x86_64.whl -w dist/
# twine upload dist/*manylinux*.whl dist/*.tar.gz

# PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring twine upload dist/*0.0.9*manylinux*.whl dist/*0.0.9.tar.gz