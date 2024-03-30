# Use Python 3.10 base image
FROM python:3.10

# Set environment variables for better performance
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set up working directory
WORKDIR /workdir

# Copy the wheel file into the container
COPY dist/*.whl .

# Install the package and remove pip cache
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ --no-cache-dir *.whl ai2-kit[all] && \
    rm -rf /root/.cache/pip/* && \
    rm -rf *.whl