FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
