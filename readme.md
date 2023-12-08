# Kaggle stanford-ribonanza-rna-folding

Competition: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding

The tokenizer and the model config are taken from https://github.com/biomed-AI/SpliceBERT/tree/main  
During my initial experiments this model performed better (or at least was converging faster)
than the model having frozen sinusoidal absolute embeddings.  
However, after switching the model to relative embeddings and training it from scratch it outperformed the pretrained model. I gave up on using pretrained weights but continued using the config and the tokenizer.

The solution and the docker image are a bit cumbersome. Some packages are never used in the final solution. The docker image size is > 20 gb. There are definitely optimizations to be made. But since this is a 'dev' image, I believe it is ok.

Build and run while bind mounting to /home/rapids/notebooks/host
```bash
docker run --gpus all --rm -it --user 1000 \
    --shm-size=10g --ulimit memlock=-1 \
    -p 0.0.0.0:6006:6006 -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    -v "$(pwd)":/home/rapids/notebooks/host \
    -v ~/.cache/huggingface:/home/rapids/.cache/huggingface \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
    -e KAGGLE_KEY=$KAGGLE_KEY \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e HF_DATASETS_IN_MEMORY_MAX_SIZE=0 \
    $(docker build --build-arg user_id=$(id -u) -q .)
```
Connect to http://localhost:8888/

