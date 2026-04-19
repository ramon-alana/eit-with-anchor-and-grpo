FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# 1. Environment Variables
ENV SSH_PORT=22 \
    UV_SYSTEM_PYTHON="true" \
    UV_NO_CACHE="true" \
    UV_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple/" \
    UV_PYTHON_INSTALL_MIRROR="https://registry.npmmirror.com/-/binary/python-build-standalone" \
    HF_HOME=/WEIGHTS \
    TORCH_HOME=/WEIGHTS

# 2. System Dependencies (APT)
# Merge sources list update, install tools, ssh, and libsuitesparse
RUN printf "deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse\n" \
> /etc/apt/sources.list && \
    apt-get update && \
    apt install -y --no-install-recommends \
    make curl openssh-server openssh-client ssh iputils-ping git net-tools wget tmux vim \
    libsuitesparse-dev && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd

# 3. Install UV
RUN pip install --no-cache-dir uv -i ${UV_INDEX_URL}

# 4. Install Python Packages
# Install setuptools first as per instructions
RUN uv pip install "setuptools<75.3.0" cython  -i ${UV_INDEX_URL}

# Copy requirements file
COPY req.txt /tmp/req.txt
# # Configure Git to use a mirror for GitHub (fixes connection refused in China)
# RUN git config --global url."https://ghproxy.com/https://github.com".insteadOf "https://github.com"
# Install requirements
RUN uv pip install -r /tmp/req.txt --system --no-build-isolation  && \
    rm /tmp/req.txt 

# Install specific versions with custom index or flags
RUN uv pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121 
RUN uv pip install trl==0.22.0 transformers==4.36.1 ml_collections --system --no-deps 

# 5. SSH Configuration
RUN echo "root:111" | chpasswd && \
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM yes" >> /etc/ssh/sshd_config && \
    printf '\n# --- fix docker exec cwd / prompt mismatch ---\n\
    if [ -n "$PWD" ] && [ -d "$PWD" ]; then\n\
        cd "$PWD" >/dev/null 2>&1\n\
    fi\n' >> /root/.bashrc

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]