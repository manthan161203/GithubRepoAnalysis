import re
import fitz  # PyMuPDF
import docx
import io

def extract_github_url_from_document(file_content: bytes, filename: str) -> str:
    """
    Extracts text from a PDF or DOCX file and returns the first GitHub repository URL found.
    Raises ValueError if no GitHub URL is found or if the file type is unsupported.
    """
    text = ""
    lower_filename = filename.lower()
    
    if lower_filename.endswith('.pdf'):
        try:
            # Open the PDF from bytes
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            pdf_document.close()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF document: {e}")
            
    elif lower_filename.endswith('.docx'):
        try:
            # Open the DOCX from bytes
            doc = docx.Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX document: {e}")
            
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
        
    # Regex to find a GitHub URL
    # Matches https://github.com/username/repo (with optional .git)
    github_pattern = r"(https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"
    match = re.search(github_pattern, text)
    
    if match:
        url = match.group(1)
        # Clean up any trailing typical punctuation that might get caught
        url = url.rstrip(').,;\'"')
        if url.endswith('.git'):
            url = url[:-4]
        return url
        
    raise ValueError("No GitHub repository URL found in the provided document.")
