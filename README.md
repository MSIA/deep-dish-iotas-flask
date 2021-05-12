# Style Transform

### Get started

Install dependencies:

`pip install -r requirements.txt`
  
Run locally:

`python app.py`

#### Docker

You may also run this app in a Docker container.

Option 1: Download a prebuilt image:

```bash
docker run --rm -p 80:80 brianrice2/deep-dish-style-transfer
```

Option 2: Build the most current version from this repo:

```bash
docker build -t deepdish .
docker run --rm -p 80:80 deepdish
```

### References

- https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask

- https://roytuts.com/upload-and-display-image-using-python-flask/