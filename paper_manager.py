#!/usr/bin/env python3
"""
Paper Management System
用于上传和记录论文的工具
"""

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
import argparse


class PaperManager:
    """管理论文的上传和记录"""
    
    def __init__(self, papers_dir="papers", records_dir="records"):
        self.papers_dir = Path(papers_dir)
        self.records_dir = Path(records_dir)
        self.records_file = self.records_dir / "papers.json"
        
        # 确保目录存在
        self.papers_dir.mkdir(exist_ok=True)
        self.records_dir.mkdir(exist_ok=True)
        
        # 加载或初始化记录
        self.papers = self._load_records()
    
    def _load_records(self):
        """加载论文记录"""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 无法加载记录文件: {e}")
                print("将创建新的记录文件")
                return []
        return []
    
    def _save_records(self):
        """保存论文记录"""
        try:
            # 先写入临时文件，确保完整性
            temp_file = self.records_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.papers, f, ensure_ascii=False, indent=2)
            # 成功后替换原文件
            temp_file.replace(self.records_file)
        except IOError as e:
            print(f"错误: 无法保存记录文件: {e}")
            raise
    
    def _calculate_md5(self, filepath):
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def upload_paper(self, file_path, title, authors=None, year=None, 
                    conference=None, notes=None, tags=None):
        """
        上传论文文件并记录元数据
        
        Args:
            file_path: 论文文件路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            conference: 会议/期刊名称
            notes: 备注
            tags: 标签列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 计算文件哈希
        file_hash = self._calculate_md5(file_path)
        
        # 检查是否已存在
        for paper in self.papers:
            if paper['md5'] == file_hash:
                print(f"警告: 该论文已存在 (ID: {paper['id']})")
                return paper['id']
        
        # 生成新的ID (确保即使删除记录后也不会冲突)
        paper_id = max([p['id'] for p in self.papers], default=0) + 1
        
        # 复制文件到papers目录
        file_extension = file_path.suffix
        new_filename = f"{paper_id}_{file_path.name}"
        destination = self.papers_dir / new_filename
        shutil.copy2(file_path, destination)
        
        # 创建记录
        record = {
            'id': paper_id,
            'title': title,
            'authors': authors or [],
            'year': year,
            'conference': conference,
            'filename': new_filename,
            'original_filename': file_path.name,
            'file_extension': file_extension,
            'md5': file_hash,
            'upload_date': datetime.now().isoformat(),
            'notes': notes,
            'tags': tags or []
        }
        
        self.papers.append(record)
        self._save_records()
        
        print(f"✓ 论文已上传成功!")
        print(f"  ID: {paper_id}")
        print(f"  标题: {title}")
        print(f"  文件: {new_filename}")
        
        return paper_id
    
    def list_papers(self, search=None, tag=None):
        """
        列出所有论文
        
        Args:
            search: 搜索关键词（在标题、作者、会议中搜索）
            tag: 按标签过滤
        """
        papers = self.papers
        
        # 过滤
        if search:
            search = search.lower()
            papers = [p for p in papers if 
                     search in p['title'].lower() or
                     search in str(p['authors']).lower() or
                     search in str(p.get('conference', '')).lower()]
        
        if tag:
            papers = [p for p in papers if tag in p.get('tags', [])]
        
        if not papers:
            print("没有找到论文")
            return
        
        print(f"\n共找到 {len(papers)} 篇论文:\n")
        print(f"{'ID':<5} {'标题':<40} {'作者':<30} {'年份':<6}")
        print("-" * 85)
        
        for paper in papers:
            paper_id = paper['id']
            title = paper['title'][:38] + '..' if len(paper['title']) > 40 else paper['title']
            authors = ', '.join(paper['authors'][:2]) if paper['authors'] else 'N/A'
            if len(paper['authors']) > 2:
                authors += f" 等 ({len(paper['authors'])}人)"
            authors = authors[:28] + '..' if len(authors) > 30 else authors
            year = paper.get('year') or 'N/A'
            
            print(f"{paper_id:<5} {title:<40} {authors:<30} {year:<6}")
    
    def get_paper(self, paper_id):
        """获取论文详细信息"""
        for paper in self.papers:
            if paper['id'] == paper_id:
                return paper
        return None
    
    def show_paper_details(self, paper_id):
        """显示论文详细信息"""
        paper = self.get_paper(paper_id)
        
        if not paper:
            print(f"未找到ID为 {paper_id} 的论文")
            return
        
        print("\n" + "=" * 60)
        print(f"论文详细信息 (ID: {paper['id']})")
        print("=" * 60)
        print(f"标题:     {paper['title']}")
        print(f"作者:     {', '.join(paper['authors']) if paper['authors'] else 'N/A'}")
        print(f"年份:     {paper.get('year', 'N/A')}")
        print(f"会议/期刊: {paper.get('conference', 'N/A')}")
        print(f"文件名:   {paper['filename']}")
        print(f"上传时间: {paper['upload_date']}")
        if paper.get('tags'):
            print(f"标签:     {', '.join(paper['tags'])}")
        if paper.get('notes'):
            print(f"备注:     {paper['notes']}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='论文管理系统 - 上传和记录论文',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传论文
  python paper_manager.py upload paper.pdf -t "深度学习综述" -a "张三" "李四" -y 2023
  
  # 列出所有论文
  python paper_manager.py list
  
  # 搜索论文
  python paper_manager.py list -s "深度学习"
  
  # 查看论文详情
  python paper_manager.py show 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # upload命令
    upload_parser = subparsers.add_parser('upload', help='上传论文')
    upload_parser.add_argument('file', help='论文文件路径')
    upload_parser.add_argument('-t', '--title', required=True, help='论文标题')
    upload_parser.add_argument('-a', '--authors', nargs='+', help='作者列表')
    upload_parser.add_argument('-y', '--year', type=int, help='发表年份')
    upload_parser.add_argument('-c', '--conference', help='会议/期刊名称')
    upload_parser.add_argument('-n', '--notes', help='备注')
    upload_parser.add_argument('--tags', nargs='+', help='标签')
    
    # list命令
    list_parser = subparsers.add_parser('list', help='列出论文')
    list_parser.add_argument('-s', '--search', help='搜索关键词')
    list_parser.add_argument('--tag', help='按标签过滤')
    
    # show命令
    show_parser = subparsers.add_parser('show', help='显示论文详情')
    show_parser.add_argument('id', type=int, help='论文ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = PaperManager()
    
    if args.command == 'upload':
        try:
            manager.upload_paper(
                file_path=args.file,
                title=args.title,
                authors=args.authors,
                year=args.year,
                conference=args.conference,
                notes=args.notes,
                tags=args.tags
            )
        except Exception as e:
            print(f"错误: {e}")
    
    elif args.command == 'list':
        manager.list_papers(search=args.search, tag=args.tag)
    
    elif args.command == 'show':
        manager.show_paper_details(args.id)


if __name__ == '__main__':
    main()
