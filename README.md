# BigData

## Requirements
### Model Download
To run our model, you will need to download the files in this google drive folder: [Model folder](https://drive.google.com/drive/folders/1oz2Jd2Mz7uHh82SrrkYeMrmesfW4QYQ2?usp=sharing). Then after the files have been installed put them in the `./models/resources ` directory. The other models are automatically downloaded in code or are on api calls.

### Environment
We recommend setting up an environment in python with venv to run our code and install the pip packages needed this can be done with the following lines of code (except windows):
```
python3 -m venv env
```
```
source ./env/bin/activate
```
```
pip install -r requirements.txt
```
Otherwise, you are welcome to read the modules in requirements.txt and install manually.

## Running the App

To run our app, run the following line of code from the main directory of the project

```
python -m streamlit run ui/app.py
```
The first time the applicaton is started the BLIP and GIT models will be downloaded. This will take some time. After they are done they get cached in a directory named hugging face in the home directory on linux so that the start up next time is quicker. Each run of the model uses a free API key that we get from replicate that runs out every so often. One key is in this repo, but if it runs out, go to the following link to make an account with a free token: [Replicate](https://replicate.com). Then replace the `replicate_api_key` variable in `/ui/app.py` to your new key.
