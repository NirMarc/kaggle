#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

echo "Downloading Spaceship Titanic dataset from Kaggle..."
mkdir -p ./data # Create data directory if it doesn't exist
kaggle competitions download -c spaceship-titanic -p ./data # Download dataset -p specifies the path
unzip ./data/spaceship-titanic.zip -d ./data # Unzip the downloaded file into the data directory -d specifies the destination destination
rm ./data/spaceship-titanic.zip
echo "Download complete. Files are saved in the ./data directory."
ls ./data