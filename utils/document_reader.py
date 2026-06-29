import pdfplumber


def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        return text.strip()

    except Exception as e:
        return f"PDF reading error: {e}"


def extract_text_from_txt(uploaded_file):
    try:
        bytes_data = uploaded_file.read()
        text = bytes_data.decode("utf-8", errors="ignore")
        return text.strip()

    except Exception as e:
        return f"TXT reading error: {e}"


def split_text_into_chunks(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks