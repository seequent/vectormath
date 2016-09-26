
.PHONY: install publish docs coverage lint graphs tests

install:
	python setup.py install

publish:
	python setup.py sdist upload

docs:
	cd docs && make html

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=vectormath --cover-html
	open cover/index.html

lint:
	pylint --output-format=html vectormath > pylint.html

graphs:
	pyreverse -my -A -o pdf -p vectormathpy vectormath/**.py vectormath/**/**.py

tests:
	nosetests --logging-level=INFO
