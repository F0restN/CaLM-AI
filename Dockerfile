# 使用官方Python基础镜像
FROM python:3.12

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露FastAPI端口
EXPOSE 8000

# 运行 site-package-check.py 设置环境
RUN python site-package-check.py

ENV PGVECTOR_CONN="postgresql+psycopg://calmadrduser:HelloWeMeetAgain#1020@host.docker.internal:5432/calmadrddb"
ENV OLLAMA_HOST="host.docker.internal"

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 