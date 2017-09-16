# mlal-webAPI


## Requirements
* Python 3

* Linux, Windows or Mac OSX

* The source code this library 

* Docker & docker-compose



## Start API

* We recommend using the `docker-compose` command for running an instance of this API.

     To start an instance of this API using the following command to run it in docker:

    `docker-compose up --build --remove-orphans`

    If you would like to run it background, use:

    `docker-compose up --build -d --remove-orphans` 
    
* If you do not wish to run it in docker (why?), please do the following
        
        virtualenv -p python3 flask
        source flask/bin/activate
        pip3 install -r requirements.txt
        python3 api.py
    
    You also need to change the server address for MongoDB, redis, and MySQL in utils.py, api.py and utils.py respectively to the address where these instances are hosted.
    
    In addition to this, you will need to run redis-server in a new terminal window using: `redis-server`
    
    And run a celery worker to perform background tasks using:
       
        virtualenv -p python3 flask
        source flask/bin/activate
        celery worker -A api.celery -l info
    

(See, it is easier with docker-compose)
        

You can then send requests to `http://localhost/sphere/api/v1/...`

See API Endpoints for more details on endpoints
