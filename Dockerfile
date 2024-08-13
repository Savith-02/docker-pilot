# Use a base image
FROM python:3.7-slim

# Set the working directory
WORKDIR /app

# Copy the rest of your application files
COPY . .

# Copy package files and install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 3000

# Define the command to run your app
CMD ["python", "main.py"]