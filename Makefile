install:
# 	python3 -m venv venv
# 	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup_full.txt

bot: 
	mkdir -p logging
	rm	-f logging/out.txt
	touch logging/out.txt
	python -u src/deploy.py 2>&1 | tee logging/out.txt
	
botcl:	
	chainlit run src/deploy.py -w --port 8080

data:
	python3 src/ingest.py

list: #List all the running bots 
	ps aux | grep python
# To kill, use command (PID is the process ID): kill PID
kill:
	pkill -f "make bot"