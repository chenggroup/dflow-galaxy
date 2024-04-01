# Use Python 3.10 base image

FROM link89/dflow-galaxy:0.1.2-launch_app-ea47760
# FROM python:3.10

# # Set environment variables for better performance
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# Set up working directory
WORKDIR /workdir

# Copy launch_app folder into the container
COPY launch_app launch_app

# Copy the wheel file into the container
COPY dist/*.whl .

# Install the package and remove pip cache
RUN pip uninstall -y dflow-galaxy || true && \
    pip install --no-cache-dir *.whl ai2-kit[all] && \
    rm -rf /root/.cache/pip/* && \
    rm -rf *.whl
