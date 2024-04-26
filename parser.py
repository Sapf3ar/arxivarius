import PyPDF2
import logging
from thefuzz import fuzz
from refextract import extract_references_from_file, extract_references_from_url
import arxiv


def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text

def parse_links(path):
    logging.warning(path)
    if "pdf" in path and "http" not in path:
        reference = extract_references_from_file(str(path))
        return reference
    elif "http" in path:
        reference = extract_references_from_url(path)
        return reference
    raise ValueError(f"Link is broken {path}")

def get_correct_name_from_topk(titles:list[str], label:str) -> int:
    # returns index of top match
    matches = [fuzz.ratio(i, label) for i in titles]
    return matches.index(max(matches))

def get_download_id(ref_q):
    if "url" in ref_q.keys():
        rref = ref_q["url"][0] 
        id_ = rref.split("/")[-1]
        return id_
    else:
        client = arxiv.Client()
        if "author" in ref_q.keys():
            query = ref_q['author'][0]

        elif "misc" in ref_q.keys():
            query = ref_q['misc'][0]
        else:
            raise ValueError("Wrong reference types, nor author field, nor title field were found.")

        logging.warning("QUERY: " + query)
        search = arxiv.Search(
              query = query,  
              max_results = 5,
              sort_by = arxiv.SortCriterion.Relevance
            )
        results = client.results(search)
        ids = []
        titles = [] 
        for r in results:
            ids.append(r.get_short_id())
            titles.append(r.title)
        try:
            best_id = ids[get_correct_name_from_topk(titles, ref_q["misc"][0])]
            return best_id
        except KeyError:
            return None
    

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    refs = parse_links(path)
    search_res = download_links(refs)
