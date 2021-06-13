## How to run

```bash
  # install packages
  $ sudo yum install git
  $ sudo yum install docker
  $ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
  
  # assign permission
  $ sudo chmod +x /usr/local/bin/docker-compose
  $ sudo usermod -aG docker $USER
  $ newgrp docker
  $ sudo systemctl restart docker
  
  # pull repo
  $ git clone https://github.com/cheat-tos/serving.git
  
  # execute
  $ cd serving
  $ docker-compose up
```

## envirment
```
system-release-2-13.amzn2.x86_64
```
