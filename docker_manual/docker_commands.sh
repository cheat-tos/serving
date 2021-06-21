# "CONFIG" settings should be customized before dockerizing.
# After configure your options, keep this folder out from tracking by using git command in below reference.
# https://stackoverflow.com/questions/10755655/git-ignore-tracked-files

# Find the local path of the latest version BentoService saved bundle
bento_service_class_name="CONFIG" # class name in service.py, which would be packed
saved_path=$(bentoml get ${bento_service_class_name}:latest --print-location --quiet)
docker_image_name="CONFIG" # name for built docker image
user_name="CONFIG" # your docker hub account name

# Build docker image using saved_path directory as the build context, replace the
docker build -t $user_name/$docker_image_name $saved_path

# Run a container with the docker image built and expose port 5000(default, you can customize)
# docker run -p 5000:5000 $user_name/$docker_image_name

# Push the docker image to docker hub for deployment
docker push $user_name/$docker_image_name
