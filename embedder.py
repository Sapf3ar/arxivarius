import json

class TermDatabase:
    def __init__(self, filepath: str):
        self.terms = self.load_terms(filepath)

    def load_terms(self, filepath: str) -> dict:
        with open(filepath, 'r') as file:
            return json.load(file)

    def get_definition(self, term: str) -> str:
        return self.terms.get(term, 'Definition not found.')


def load_term_database() -> TermDatabase:
    return TermDatabase('path/to/terms.json')
