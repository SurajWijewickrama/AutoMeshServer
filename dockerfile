# Use an official Python runtime as a base image
FROM python:3.9-slim

# Create and switch to a working directory
WORKDIR /app

# Copy only the requirements file first (for efficient caching)
COPY AutoMeshAPI/requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your source code
COPY AutoMeshAPI/ /app

# Expose the port that your API listens on (Cloud Run expects port 8080 by default)
EXPOSE 8080

# Set the command to run your application
# If "main.py" is your entry point, use it here:
CMD ["python", "main.py"]
