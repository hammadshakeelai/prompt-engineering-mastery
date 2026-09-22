import arxiv
import os
import json

def download_papers(queries, max_results=5, output_dir="obsidian_vault/papers_pdf", metadata_dir="obsidian_vault/papers_metadata"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    client = arxiv.Client()
    
    for topic, query_string in queries.items():
        print(f"Searching arXiv for: {topic} ({query_string})")
        search = arxiv.Search(
            query=query_string,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        metadata_list = []
        for result in client.results(search):
            clean_id = result.get_short_id()
            print(f"  Found: {clean_id} - {result.title[:50]}...")
            
            # Save PDF
            pdf_path = os.path.join(output_dir, f"{clean_id.replace('.', '_')}.pdf")
            if not os.path.exists(pdf_path):
                print(f"    Downloading PDF...")
                try:
                    result.download_pdf(dirpath=output_dir, filename=f"{clean_id.replace('.', '_')}.pdf")
                except Exception as e:
                    print(f"    Failed to download: {e}")
            
            # Save Metadata
            metadata_list.append({
                "id": clean_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "summary": result.summary,
                "published": str(result.published),
                "pdf_url": result.pdf_url
            })
            
            # Create Obsidian Note
            md_path = os.path.join("obsidian_vault", f"{clean_id.replace('.', '_')}.md")
            with open(md_path, 'w', encoding='utf-8') as md:
                md.write(f"---\n")
                md.write(f"arxiv_id: {clean_id}\n")
                md.write(f"title: \"{result.title}\"\n")
                md.write(f"authors: {', '.join([a.name for a in result.authors])}\n")
                md.write(f"published: {str(result.published)}\n")
                md.write(f"tags: [paper, {topic}]\n")
                md.write(f"---\n\n")
                md.write(f"# {result.title}\n\n")
                md.write(f"## Abstract\n{result.summary}\n\n")
                md.write(f"## Links\n- [[papers_pdf/{clean_id.replace('.', '_')}.pdf]]\n")
                md.write(f"- [arXiv Link]({result.entry_id})\n")

        # Save query metadata
        with open(os.path.join(metadata_dir, f"{topic}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2)

if __name__ == "__main__":
    queries = {
        "textgrad": 'all:"TextGrad" AND all:"Prompt"',
        "dspy": 'all:"DSPy" AND all:"declarative"',
        "prompt_optimization": 'all:"PromptBreeder" OR all:"Optimization by PROmpting"',
    }
    print("Starting Open-Source Paper Downloader...")
    download_papers(queries, max_results=3)
    print("Finished downloading research papers.")
