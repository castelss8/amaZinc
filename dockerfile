# Indicate the Gurobi reference image
FROM gurobi/optimizer:latest
FROM gurobi/python:latest as gurobi
FROM minizinc/minizinc:latest

# Set the application directory
WORKDIR /src

# Copy the source
COPY . .

# Installing python
RUN apt-get update \
  && apt-get install -y python3 \
  && apt-get install -y python3-pip \
  && apt-get install -y python3-venv \
  && apt-get install -y build-essential \
  && apt-get install -y python3-dev \
  && apt-get install -y libffi-dev

# Create a virtual environment and activate it
RUN python3 -m venv venv
ENV PATH="/src/venv/bin:$PATH"
ENV GRB_LICENSE_FILE="/src/gurobi.lic"

ADD requirements.txt ./src/requirements.txt
RUN pip3 install --no-cache-dir -v -r requirements.txt

# Command used to start the application
#CMD ["python","main_solutions.py"]