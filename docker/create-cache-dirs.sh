#!/bin/bash
# create-cache-dirs.sh

echo "创建缓存目录..."

# 从 .env 文件读取缓存目录路径
source .env

# 创建所有缓存目录
mkdir -p ${CACHE_DIR_PIP}
mkdir -p ${CACHE_DIR_PIP_WHEELS}
mkdir -p ${CACHE_DIR_UV}
mkdir -p ${CACHE_DIR_APT}

# 设置权限（确保容器可以写入）
chmod 777 ${CACHE_DIR_PIP} 2>/dev/null || true
chmod 777 ${CACHE_DIR_PIP_WHEELS} 2>/dev/null || true
chmod 777 ${CACHE_DIR_UV} 2>/dev/null || true
chmod 777 ${CACHE_DIR_APT} 2>/dev/null || true

echo "缓存目录创建完成："
echo "  PIP缓存: ${CACHE_DIR_PIP}"
echo "  PIP Wheels: ${CACHE_DIR_PIP_WHEELS}"
echo "  UV缓存: ${CACHE_DIR_UV}"
echo "  APT缓存: ${CACHE_DIR_APT}"