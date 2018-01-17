export VERSION=`python setup.py --version 2>/dev/null`

tag:
	git commit -a -m "Release $(VERSION)"; true
	git tag v$(VERSION)
	git push origin --all
	git push origin --tags
	twine upload dist/*

all:
	rm -rf dist
	pip install wheel -U
	python3 setup.py register
	python3 setup.py bdist_wheel
	python3 setup.py sdist --formats=zip
	twine upload dist/*

