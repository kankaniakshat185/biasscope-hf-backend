FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Prisma relies on an internal Node.js runtime which throws libatomic.so.1 errors on Debian 'slim' images
# (tesseract-ocr used to be installed here for an OCR fallback path that
# was removed from the product — see AUDIT_TASKS.md X1)
RUN apt-get update -y && apt-get install -y libatomic1 openssl

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -m spacy download en_core_web_sm

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

# Schema sync now runs once here, at container start, instead of inside
# app/main.py's startup handler. `&&` means a failed sync stops the
# container from ever serving traffic on a half-synced schema, instead of
# an os.system() call inside the app silently swallowing the exit code.
CMD ["sh", "-c", "python3 -m prisma db push --skip-generate && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
