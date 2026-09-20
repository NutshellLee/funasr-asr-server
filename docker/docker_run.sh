export MY_CONTAINER="funasr_docker"
# 初始设置
setup_path=$(dirname $(cd `dirname $0`;pwd))
num=`docker ps -a|grep "$MY_CONTAINER" | wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
    docker run -it --name $MY_CONTAINER --gpus '"device=0,1"' \
        -p 0.0.0.0:8081:8081 \
        funasr_docker:3.2
else
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER bash
fi

# Mount the project from the host (defaults to the current working directory):
# -v "$(pwd)":/root/asr \

# python fast_api_server.py
