#!/bin/bash
set -e

# Generate Python wheel package
rm dist/* || true
poetry build

# Get short Git hash
VERISON=$(poetry version --no-ansi | awk '{print $2}')
GIT_HASH=$(git rev-parse --short HEAD)

TAG=dflow-galaxy:$VERISON-$GIT_HASH

docker build -t $TAG .

echo "Run the following command to push the image to Docker Hub:"
echo "docker push $TAG"
