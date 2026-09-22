import json
import os
import urllib.request
import time

METADATA_DIR = r"C:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\papers_metadata"
PDF_DIR = r"C:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\papers_pdf"
VAULT_DIR = r"C:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault"

def download_pdf(url, output_path):
    # arXiv PDF URLs usually look like http://arxiv.org/pdf/1706.03762v5
    if url.startswith("http://"):
        url = url.replace("http://", "https://")
    if not url.endswith(".pdf"):
        url += ".pdf"
        
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    downloaded_ids = set()
    
    for filename in os.listdir(METADATA_DIR):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(METADATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                papers = json.load(f)
            except:
                continue
                
        print(f"Processing {filename}")
        for paper in papers[:4]:  # Top 4 per category
            arxiv_id = paper.get('id')
            if not arxiv_id or arxiv_id in downloaded_ids:
                continue
                
            clean_id = arxiv_id.replace('.', '_')
            pdf_path = os.path.join(PDF_DIR, f"{clean_id}.pdf")
            pdf_url = paper.get('pdf_url')
            
            if pdf_url and not os.path.exists(pdf_path):
                print(f"  Downloading {arxiv_id}...")
                success = download_pdf(pdf_url, pdf_path)
                if success:
                    # Create markdown stub
                    md_path = os.path.join(VAULT_DIR, f"{clean_id}.md")
                    with open(md_path, 'w', encoding='utf-8') as md:
                        md.write(f"---\n")
                        md.write(f"arxiv_id: {arxiv_id}\n")
                        md.write(f"title: \"{paper.get('title')}\"\n")
                        md.write(f"authors: {', '.join(paper.get('authors', []))}\n")
                        md.write(f"published: {paper.get('published')}\n")
                        md.write(f"---\n\n")
                        md.write(f"# {paper.get('title')}\n\n")
                        md.write(f"## Abstract\n{paper.get('summary')}\n\n")
                        md.write(f"## Links\n- [[papers_pdf/{clean_id}.pdf]]\n")
                        md.write(f"- [arXiv Link]({pdf_url.replace('.pdf', '')})\n")
                time.sleep(3) # Respect arXiv limits
            downloaded_ids.add(arxiv_id)

if __name__ == "__main__":
    main()
