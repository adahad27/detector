#Using a slim version of python for less disk consumption
FROM python:3.10.18-slim

#Set the working directory to be the /app directory
WORKDIR /app

#Copy over requirements.txt so we can run pip install
COPY requirements.txt requirements.txt

#Updating pip before running it
RUN pip install --upgrade pip

#Run the pip install
RUN pip install -r requirements.txt

#This should copy all of the source files to the current directory
COPY dataset.py metadata.py model.py testing.py training.py model_parameters.pt ./

CMD [ "python3", "testing.py" ]