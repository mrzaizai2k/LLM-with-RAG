install:
# 	python3 -m venv venv
# 	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt

bot: 
	chainlit run src/deploy.py -w

data:
	python3 src/ingest.py