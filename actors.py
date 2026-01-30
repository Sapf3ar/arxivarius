class Researcher:
    def request_information(self, term: str) -> str:
        return "Searching for term: " + term

class Professor:
    def evaluate_response(self, response: str) -> bool:
        # Implement evaluation logic
        return True
