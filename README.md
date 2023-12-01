# BigData

## Setup
```
python3 -m venv env &&
source ./env/bin/activate &&
pip install -r requirements.txt
```
## Run app

Run from root directory

```
python -m streamlit run ui/app.py
```

## Docker Setup

1. To build the docker image, run the build.sh script

    ```
    sh build.sh
    ```
2. To deploy the image locally, run the deploy-local.sh script
    ```
    sh deploy-local.sh
    ```
3. To cleanup everything, run the cleanup.sh script
    ```
    sh cleanup.sh
    ```
