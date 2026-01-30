import pytest
from main import WorkFlow, init_keys, make_file_upload, download_papers

# Test class for WorkFlow functionalities
class TestWorkFlow:
    def setup_method(self):
        self.workflow = WorkFlow()

    def test_init_keys(self):
        # Test initialization of session state keys
        init_keys()
        assert "messages" in st.session_state
        assert "file_uploaded" in st.session_state

    def test_make_file_upload(self):
        make_file_upload()
        assert st.session_state.file_uploaded is True

    def test_download_papers_valid_link(self):
        # Mock a valid link to a paper
        valid_link = "https://arxiv.org/abs/1234.5678"
        papers = download_papers([valid_link])
        assert len(papers) > 0

    def test_download_papers_invalid_link(self):
        # Mock an invalid link to a paper
        invalid_link = "https://arxiv.org/abs/invalid"
        papers = download_papers([invalid_link])
        assert papers is None

    # Add more tests for other functionalities and edge cases

if __name__ == "__main__":
    pytest.main()