ARG VARIANT=bullseye
ARG VERSION=3.12
FROM --platform=amd64 mcr.microsoft.com/devcontainers/python:${VERSION}-${VARIANT}

RUN export DEBIAN_FRONTEND=noninteractive \
    && curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc \
    && curl https://packages.microsoft.com/config/debian/11/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list\
    && apt-get update \
    && apt-get install -y xdg-utils unixodbc \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && curl -fsSL https://aka.ms/install-azd.sh | bash \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*