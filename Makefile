user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
# phism=/workspace
vhls=/data/Vivado

# docker buildx pruney

# Build docker container
build-docker: 
	(docker build --build-arg VHLS_PATH=$(vhls) . --tag pom_dev)

# Enter docker container
shell: build-docker
	docker run -it -v $(shell pwd):/home/workspace -v $(vhls):$(vhls) --name pom_dev pom_dev:latest /bin/bash 

# docker exec -it -v /home/wczhang/Xilinx:/home/wczhang/Xilinx flowgnn /bin/bash 


