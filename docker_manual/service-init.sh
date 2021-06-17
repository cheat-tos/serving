# Initialize Docker Swarm Cluster(& Manager Node)
docker swarm init
# Run specified Services(each Service has multiple container replicas) in <docker-compose.yml> as name <dkt-service>
docker stack deploy -c docker-compose.yml dkt-service

# Running Services in Dockker Swarm Cluster
echo "All Services are running ... Below is Running Services which current manage node tracks."
docker service ls

# You can get Service logs by executing next line.
# docker service logs -f <SERVICE-NAME>

# You can update(rolling-update) service with new image by executing next line.
#docker service update \
#  --update-parallelism 1 \
#  --update-delay 10s \
#  --image kpic5014/dkt-client-flask:latest \
#  --detach=false \
#  dkt-service_client
