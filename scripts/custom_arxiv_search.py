import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os

def search_arxiv(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
    except Exception as e:
        print(f"Failed to fetch from arXiv: {e}")
        return []
        
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    results = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        published = entry.find('atom:published', ns).text
        paper_id = entry.find('atom:id', ns).text.split('/')[-1]
        
        pdf_url = ""
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_url = link.attrib.get('href')
                break
                
        results.append({
            'id': paper_id,
            'title': title,
            'summary': summary,
            'authors': authors,
            'published': published,
            'pdf_url': pdf_url
        })
        
    return results

def main():
    queries = {
        "prompt_engineering": "prompt engineering",
        "cot": "chain of thought prompting",
        "tot": "tree of thoughts",
        "dspy": "DSPy declarative",
        "prompt_injection": "prompt injection LLM"
    }
    
    out_dir = r"C:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\papers_metadata"
    os.makedirs(out_dir, exist_ok=True)
    
    for name, q in queries.items():
        print(f"Searching for {q}...")
        data = search_arxiv(q, max_results=5)
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
if __name__ == "__main__":
    main()
