all:
	python setup.py register
	python setup.py sdist --formats=gztar,zip upload

install:
	python setup.py install
	cd calysto/language/scheme; python setup.py install

install3:
	python3 setup.py install
	cd calysto/language/scheme; python3 setup.py install
