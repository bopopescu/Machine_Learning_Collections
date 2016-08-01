# Hello World to Tensorflow with Docker & Ipython


## Overview

Running these types of projects on local computers can be very difficult sometimes. So, we will use Docker instead.


### Installing Docker
For this part, please follow Docker manual. [Get Started with Docker](http://docs.docker.com/mac/started/)

### Running Docker Daemon
Open a terminal and start by typing..

	docker-machine start default
	eval "$(docker-machine env default)"

### Checking Docker IP
	docker-machine ls
	
This is the response I get from the command

NAME      ACTIVE   DRIVER       STATE     URL                         SWARM

default   *        virtualbox   Running   tcp://192.168.99.100:2376


So my Docker host IP is 192.168.99.100

### Running Tensorflow Ipython
	docker run  -d -p 5555:8888  b.gcr.io/tensorflow/tensorflow sh ./run_jupyter.sh

This part deserves some explanation. Here are the steps we have taken

1. We run docker image provided by [google](http://www.tensorflow.org/get_started/os_setup.md#docker-based_installation) by calling "docker run b.gcr.io/tensorflow/tensorflow"
2. -d command detaches the container and runs it in the background
3. -p is the port flag. Ipython runs on 8888 port. -p 8888:8888 exposes container's 8888 port to outside world from port 8888. So we use the same port for this basic example.
4. Docker image provided by Google comes with Ipython and a shell script to run it. sh ./run_jupyter.sh enables to run it.


### Run Ipython from your browser
Just go to http://192.168.99.100:8888 (DockerIP:port) and you will see Ipython is running

### Learn how Ipython works
[Ipython](http://ipython.org/)


## Accessing machine's command line 
	eval "$(docker-machine env default)"
	docker ps
	docker exec -i -t c8fc6be9a0e2 bash
