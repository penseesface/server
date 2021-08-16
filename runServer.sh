sudo docker run --gpus all --shm-size=1g --ulimit memlock=-1 --rm -it -p8000:8000 -p8001:8001 -p8002:8002  --ulimit stack=67108864 -v /home/allen/Documents/workplace/triton/server/model_repository:/models lpr_server:v2 tritonserver --model-repository=/models

 