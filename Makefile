all:
	pip install wheel -U
	python setup.py register
	python setup.py bdist_wheel
	python setup.py sdist --formats=zip
	twine upload dist/*

