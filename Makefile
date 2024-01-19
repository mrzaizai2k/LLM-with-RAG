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

list: #List all the running bots 
	ps aux | grep python
# To kill, use command (PID is the process ID): kill PID
kill:
	pkill -f "make bot"