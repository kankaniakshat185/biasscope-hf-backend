FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Prisma relies on an internal Node.js runtime which throws libatomic.so.1 errors on Debian 'slim' images
RUN apt-get update -y && apt-get install -y libatomic1 openssl

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Install the correct linux Prisma query engine binary natively inside the Docker image before starting
RUN prisma py fetch
# Force regenerate the Prisma python client specifically for Linux (overwriting the macOS version you uploaded)
RUN prisma generate

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
