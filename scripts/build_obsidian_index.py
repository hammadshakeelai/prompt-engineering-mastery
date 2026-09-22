import os
import glob

def build_index(vault_dir="obsidian_vault", repo_dir="."):
    index_path = os.path.join(vault_dir, "000_Master_Brain_Index.md")
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# 🧠 The Prompt Engineering Master Brain Index\n\n")
        f.write("This is the autonomous, auto-generated Knowledge Graph for the Prompt Engineering Mastery repository. It connects all theoretical guides, raw academic papers, test-time compute facts, and security research into a single interactive node structure.\n\n")
        
        f.write("## 📚 1. Curated Research Modules (Deep Dives)\n")
        for folder in sorted(glob.glob(f"{repo_dir}/0*")):
            if os.path.isdir(folder):
                folder_name = os.path.basename(folder)
                f.write(f"### {folder_name.replace('_', ' ').title()}\n")
                for md_file in glob.glob(f"{folder}/*.md"):
                    basename = os.path.basename(md_file)
                    f.write(f"- [[{folder_name}/{basename}]]\n")
                f.write("\n")
                
        f.write("## 🔬 2. Raw Subagent Research (Web & Factual Dumps)\n")
        raw_dir = os.path.join(vault_dir, "raw_research")
        if os.path.exists(raw_dir):
            for md_file in glob.glob(f"{raw_dir}/*.md"):
                basename = os.path.basename(md_file)
                f.write(f"- [[raw_research/{basename}]]\n")
            f.write("- [[raw_research/OBLITERATUS]] - (Abliteration & Orthogonal Direction Modification Toolkit)\n")
            f.write("- [[raw_research/chatgpt_system_prompts]] - (Extracted System Prompts for ChatGPT, Claude, Apple Intelligence)\n\n")
            
        f.write("## 📄 3. Academic Paper Vault (arXiv)\n")
        for md_file in glob.glob(f"{vault_dir}/*_*.md"):
            basename = os.path.basename(md_file)
            if basename != "000_Master_Brain_Index.md":
                f.write(f"- [[{basename}]]\n")
        
        f.write("\n## 🪝 4. Research Hooks & Meta-Prompts\n")
        for hook_file in glob.glob(f"{repo_dir}/research_engine/*.md"):
            f.write(f"- [[research_engine/{os.path.basename(hook_file)}]]\n")
            
if __name__ == "__main__":
    print("Building Obsidian Master Graph Index...")
    build_index()
    print("Brain index generated successfully at obsidian_vault/000_Master_Brain_Index.md")
