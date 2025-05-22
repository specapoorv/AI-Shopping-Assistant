# Project Details

This project is an attempt at the Appian AIgnite Hackathon, Round 2.  
It is an AI-Powered Personal Shopping Assistant for E-Commerce proof of concept on a ficticious footwear dataset.  


# Setup Instructions

Clone this repository 

Download the zip file in the following link (This is the dataset we trained on):  
https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip  

Extract it and place it inside the same working directory and rename the folder to shoes  
Run the python script shoes_init.py  
(This converts it into the format we need for the app to run)  

Create a .env file and create an entry:  
GEMINI_API_KEY=<insert your api key here>  

Install all the dependencies in requirements.txt  

Run app.py  