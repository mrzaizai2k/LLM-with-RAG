install:
# 	python3 -m venv venv
# 	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt

bot: 
	chainlit run deploy.py -w