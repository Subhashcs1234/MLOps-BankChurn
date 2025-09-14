
FROM Python:3.9

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Setting the address to 0.0.0.0 tells the server to listen on all available network interfaces. 
# server.port=8501: Sets the port to 8501 (Streamlit’s default, but explicit)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
