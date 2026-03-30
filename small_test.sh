rm -r /tmp/test
conda run --no-capture-output -n pytorch200 python -u \
    train.py --epochs 5 \
    --dataset-length 100 \
    --batch-size 8 \
    --device cuda \
    --output-dir /tmp/test

