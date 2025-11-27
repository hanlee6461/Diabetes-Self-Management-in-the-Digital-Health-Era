import os
import pymupdf4llm
import pathlib

# PDF Folder directory path
pdf_folder = r"C:\Users\ehdgh\Desktop\Programs\python\CA_TM\060325\pdf_cleaning_73"
# Markdown save directory path
markdown_folder = os.path.join(pdf_folder, "Markdown")
pathlib.Path(markdown_folder).mkdir(parents=True, exist_ok=True)

# Markdown and save
for file_name in os.listdir(pdf_folder):
    if file_name.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file_name)
        
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(markdown_folder, base_name + ".md")
        
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            pathlib.Path(output_path).write_bytes(md_text.encode("utf-8"))
            print(f"Completed: {file_name} → {base_name}.md")
        except Exception as e:
            print(f"Error: {file_name} | Reason : {e}")
