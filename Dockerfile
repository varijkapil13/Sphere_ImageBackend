FROM python:3.6
WORKDIR /usr/src/app

# install supervisord
RUN apt-get update && apt-get install -y supervisor

# copy requirements and install (so that changes to files do not mean rebuild cannot be cached)


COPY requirements.txt /usr/src/app
RUN pip3 install -r requirements.txt

# copy all files into the container
COPY . /usr/src/app

# expose port 5000 of the container
EXPOSE 5000

# needs to be set else Celery gives an error (because docker runs commands inside container as root)
ENV C_FORCE_ROOT=1

# run supervisord
CMD ["/usr/bin/supervisord"]