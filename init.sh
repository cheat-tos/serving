# install packages
sudo yum -y install git
sudo yum -y install docker
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
sudo pip3 install apache-airflow
sudo pip3 uninstall sqlalchemy -y
sudo pip3 install 'sqlalchemy < 1.4.0' apache-airflow attrdict mlflow

# assign permission
sudo chmod +x /usr/local/bin/docker-compose
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl restart docker

# set path
export AIRFLOW_HOME=/home/ec2-user/airflow

# update sqlite
sudo yum -y install wget tar gzip gcc make expect
wget https://www.sqlite.org/src/tarball/sqlite.tar.gz
tar xzf sqlite.tar.gz
cd sqlite/
export CFLAGS="-DSQLITE_ENABLE_FTS3 \
    -DSQLITE_ENABLE_FTS3_PARENTHESIS \
    -DSQLITE_ENABLE_FTS4 \
    -DSQLITE_ENABLE_FTS5 \
    -DSQLITE_ENABLE_JSON1 \
    -DSQLITE_ENABLE_LOAD_EXTENSION \
    -DSQLITE_ENABLE_RTREE \
    -DSQLITE_ENABLE_STAT4 \
    -DSQLITE_ENABLE_UPDATE_DELETE_LIMIT \
    -DSQLITE_SOUNDEX \
    -DSQLITE_TEMP_STORE=3 \
    -DSQLITE_USE_URI \
    -O2 \
    -fPIC"
export PREFIX="/usr/local"
LIBS="-lm" ./configure --disable-tcl --enable-shared --enable-tempstore=always --prefix="$PREFIX"
make
sudo make install
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
cd ..
rm -r sqlite sqlite.tar.gz

# pull repo
git clone https://github.com/cheat-tos/serving.git

# init
airflow db init
sed -i 's/load_examples = True/load_examples = False/g' airflow/airflow.cfg
mkdir airflow/dags
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# create user
airflow users create --username admin --firstname Peter --lastname Parker --password 1234 --role Admin --email spiderman@superhero.org

# run scheldurer
airflow scheduler

# run airflow gui server
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
airflow webserver -p 6006

# for docker
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
cd serving
docker-compose up