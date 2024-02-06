install:
# 	python3 -m venv venv
# 	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt

ragqa:
	python3 src/ragqa.py
bot: 
	mkdir -p logging
	rm	-f logging/out.txt
	touch logging/out.txt
	python -u src/api.py 2>&1 | tee logging/out.txt

data:
	python3 src/ingest.py

list: #List all the running bots 
	ps aux | grep python
# To kill, use command (PID is the process ID): kill PID
kill:
	pkill -f "make bot"

api:
	python3 src/api.py

test:
#	python3 src/api.py
	curl -X POST -H "Content-Type: application/json" -d '{"query": "who is karger"}' http://localhost:8083/query
# python3 src/test_api.py