#!/bin/bash
# 示例脚本：展示如何使用论文管理系统

# 上传第一篇论文
python paper_manager.py upload paper1.pdf \
    -t "深度学习在计算机视觉中的应用" \
    -a "张三" "李四" \
    -y 2023 \
    -c "CVPR" \
    --tags "深度学习" "计算机视觉"

# 上传第二篇论文
python paper_manager.py upload paper2.pdf \
    -t "Transformer模型详解" \
    -a "王五" \
    -y 2024 \
    -c "NeurIPS" \
    -n "重要参考论文" \
    --tags "Transformer" "NLP"

# 列出所有论文
python paper_manager.py list

# 搜索论文
python paper_manager.py list -s "深度学习"

# 按标签筛选
python paper_manager.py list --tag "深度学习"

# 查看论文详情
python paper_manager.py show 1
