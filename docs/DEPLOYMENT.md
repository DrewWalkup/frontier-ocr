# Deployment Guide (Ubuntu)

This guide explains how to deploy Frontier OCR as a background service on Ubuntu using systemd.

## Prerequisites

1.  **Ubuntu Server** (20.04 or later recommended)
2.  **uv** installed:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  **Project Cloned**:
    ```bash
    git clone https://github.com/your-org/frontier-ocr.git
    cd frontier-ocr
    ```
4.  **Dependencies Installed**:
    ```bash
    uv sync --extra paddle
    # Install the correct paddlepaddle or paddlepaddle-gpu runtime (see README.md)
    ```

    If you want a core-only deployment that does not serve OCR yet, install without extras. For a Paddle deployment, use the `paddle` extra and keep `OCR_ENABLED_BACKENDS=paddle`.

## Setup Systemd Service

1.  **Edit the Service File**

    Open `deployment/systemd/frontier-ocr.service` and verify the `User`, `Group`, and `WorkingDirectory` match your server setup. The default assumes a user named `ubuntu` and the project at `/home/ubuntu/frontier-ocr`.

    If your `uv` is installed in a different location (check with `which uv`), check that `ExecStart` points to the correct binary path. The default is `/home/ubuntu/.cargo/bin/uv`.

    For a Paddle-only deployment, also verify the environment includes:

    ```bash
    OCR_DEFAULT_BACKEND=auto
    OCR_ENABLED_BACKENDS=paddle
    ```

2.  **Copy to System Directory**

    ```bash
    sudo cp deployment/systemd/frontier-ocr.service /etc/systemd/system/
    ```

3.  **Reload Systemd**

    ```bash
    sudo systemctl daemon-reload
    ```

4.  **Enable and Start**

    ```bash
    # Enable to start on boot
    sudo systemctl enable frontier-ocr

    # Start immediately
    sudo systemctl start frontier-ocr
    ```

5.  **Check Status**

    ```bash
    sudo systemctl status frontier-ocr
    ```

## Logs

View logs using `journalctl`:

```bash
# View real-time logs
sudo journalctl -u frontier-ocr -f

# View last 100 lines
sudo journalctl -u frontier-ocr -n 100
```

## Updates

To deploy new code:

1.  Pull changes:
    ```bash
    git pull
    uv sync --extra paddle
    ```
2.  Restart service:
    ```bash
    sudo systemctl restart frontier-ocr
    ```
