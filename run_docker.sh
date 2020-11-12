if docker image inspect eegvip:0.1 >/dev/null 2>&1; then
echo 'container exists, not rebuilding. remove container to update'
else 
docker build --tag eegvip:0.1 .
fi 
docker run --gpus all -v $(pwd):/tf -it -p 8888:8888 -p 6006:6006 eegvip:0.1
