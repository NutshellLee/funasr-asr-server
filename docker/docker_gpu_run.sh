docker run -it --name funasr_docker --gpus '"device=0"' \
        -p 0.0.0.0:8081:8081 \
        funasr_docker:3.2
