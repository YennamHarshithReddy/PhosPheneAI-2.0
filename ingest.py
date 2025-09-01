import os
import zipfile
import json
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import yaml
import hashlib

def load_config(config_name="base"):
    """Load chunking configuration from YAML files"""
    config_file = f"configs/{config_name}.yaml"
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_html_article(file_path, filename):
    """Extract metadata and content from HTML news articles"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
        tag.decompose()
    title = ""
    if soup.find('title'):
        title = soup.find('title').get_text().strip()
    elif soup.find('h1'):
        title = soup.find('h1').get_text().strip()
    author = ""
    for selector in ['[rel="author"]', '.author', '.byline', '[itemprop="author"]']:
        author_elem = soup.select_one(selector)
        if author_elem:
            author = author_elem.get_text().strip()
            break
    pub_date = ""
    for selector in ['time', '[datetime]', '.date', '.publish-date']:
        date_elem = soup.select_one(selector)
        if date_elem:
            pub_date = date_elem.get('datetime', date_elem.get_text()).strip()
            break
    url = ""
    canonical = soup.find('link', {'rel': 'canonical'})
    if canonical:
        url = canonical.get('href', '')
    main_content = ""
    for selector in ['article', '.content', '.article-body', 'main', '.post-content']:
        content_elem = soup.select_one(selector)
        if content_elem:
            main_content = content_elem.get_text(separator=' ', strip=True)
            break
    if not main_content:
        main_content = soup.get_text(separator=' ', strip=True)
    main_content = ' '.join(main_content.split())
    return {
        'content': main_content,
        'title': title,
        'author': author,
        'publish_date': pub_date,
        'url': url,
        'source_file': filename
    }

def deduplicate_articles(articles):
    """Remove near-identical articles based on content similarity"""
    unique_articles = []
    seen_hashes = set()   
    for article in articles:
        content_hash = hashlib.md5(article['content'][:500].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_articles.append(article)
    return unique_articles

def main():   
    config = load_config("alt")  
    zip_path = "data/news/news.zip"
    extract_dir = "data/news/extracted"
    index_dir = f"faiss_index_{config['name']}"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extraction complete! Files are in {extract_dir}")
    articles = []
    for root, dirs, files in os.walk(extract_dir):
        for filename in files:
            if filename.endswith(('.html', '.htm', '.txt')):
                file_path = os.path.join(root, filename)
                try:
                    if filename.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        articles.append({
                            'content': content,
                            'title': filename.replace('.txt', ''),
                            'author': '',
                            'publish_date': '',
                            'url': '',
                            'source_file': filename
                        })
                    else:
                        articles.append(parse_html_article(file_path, filename))
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
    articles = deduplicate_articles(articles)
    print(f"Parsed {len(articles)} unique articles")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        separators=config['separators']
    )
    
    docs = []
    for article in articles:
        chunks = splitter.split_text(article['content'])
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    'source_file': article['source_file'],
                    'title': article['title'],
                    'author': article['author'],
                    'publish_date': article['publish_date'],
                    'url': article['url'],
                    'chunk_id': f"{article['source_file']}_{i}"
                }
            ))
    
    print(f"Created {len(docs)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    start_time = datetime.now()
    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    index_time = (datetime.now() - start_time).total_seconds()
    db.save_local(index_dir)
    stats = {
        'total_articles': len(articles),
        'total_chunks': len(docs),
        'index_time_seconds': index_time,
        'config_used': config
    }
    
    with open(f"{index_dir}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"FAISS index built and saved to {index_dir}")
    print(f"Index time: {index_time:.2f} seconds")

if __name__ == "__main__":
    main()
